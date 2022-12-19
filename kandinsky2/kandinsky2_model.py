from .model.model_creation import create_model, create_gaussian_diffusion
from transformers import AutoTokenizer
from PIL import Image
import cv2
import torch
from omegaconf import OmegaConf
import math
from .model.text_encoders import TextEncoder
from .vqgan.autoencoder import VQModelInterface, AutoencoderKL
from copy import deepcopy
import torch.nn.functional as F
import numpy as np
from .utils import prepare_image, q_sample, process_images, prepare_mask


class Kandinsky2:
    def __init__(self, config, model_path, device, task_type='text2img'):
        self.config = config
        self.device = device
        self.task_type = task_type
        if task_type == 'text2img' or task_type == 'img2img':
            self.config['model_config']['up'] = False
            self.config['model_config']['inpainting'] = False
        elif task_type == 'inpainting':
            self.config['model_config']['up'] = False
            self.config['model_config']['inpainting'] = True
        else:
            raise ValueError('Only text2img, img2img and inpainting is available')
        self.tokenizer1 = AutoTokenizer.from_pretrained(self.config['tokenizer_name1'])
        self.tokenizer2 = AutoTokenizer.from_pretrained(self.config['tokenizer_name2'])

        self.text_encoder1 = TextEncoder(**self.config['text_enc_params1']).to(self.device).eval()
        self.text_encoder2 = TextEncoder(**self.config['text_enc_params2']).to(self.device).eval()

        self.use_fp16 = self.config['model_config']['use_fp16']

        if self.config['image_enc_params'] is not None:
            self.use_image_enc = True
            self.scale = self.config['image_enc_params']['scale']
            if self.config['image_enc_params']['name'] == 'AutoencoderKL':
                self.image_encoder = AutoencoderKL(**self.config['image_enc_params']['params']).to(self.device)
            elif self.config['image_enc_params']['name'] == 'VQModelInterface':
                self.image_encoder = VQModelInterface(**self.config['image_enc_params']['params']).to(self.device)
            self.image_encoder.eval()
        else:
            self.use_image_enc = False
        self.config['model_config']['cache_text_emb'] = True
        self.model = create_model(**self.config['model_config'])
        self.model.load_state_dict(torch.load(model_path), strict=False)
        if self.use_fp16:
            self.model.convert_to_fp16()
            self.text_encoder1 = self.text_encoder1.half()
            self.text_encoder2 = self.text_encoder2.half()
            self.image_encoder = self.image_encoder.half()
        self.model.eval()
        self.model.to(self.device)

    def get_new_h_w(self, h, w):
        new_h = h // 64
        if h % 64 != 0:
            new_h += 1
        new_w = w // 64
        if w % 64 != 0:
            new_w += 1
        return new_h * 8, new_w * 8

    @torch.no_grad()
    def encode_text(self, text_encoder, tokenizer, prompt, batch_size):
        text_encoding = tokenizer(
            [prompt] * batch_size + [''] * batch_size,
            max_length=77,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt")

        tokens = text_encoding['input_ids'].to(self.device)
        mask = text_encoding['attention_mask'].to(self.device)

        full_emb, pooled_emb = text_encoder(tokens=tokens, mask=mask)
        return full_emb, pooled_emb

    @torch.no_grad()
    def generate_img(self, prompt, batch_size=1,
                     diffusion=None,
                     guidance_scale=7, progress=True, dynamic_threshold_v=99.5,
                     denoised_type='dynamic_threshold', init_step=None, noise=None,
                     init_img=None, img_mask=None, h=512, w=512, sampler='ddim_sampler', ddim_eta=0.8,
                     ):

        new_h, new_w = self.get_new_h_w(h, w)
        full_batch_size = batch_size * 2
        model_kwargs = {}
        if noise is not None and self.use_fp16:
            noise = noise.half()
        if init_img is not None and self.use_fp16:
            init_img = init_img.half()
        if img_mask is not None and self.use_fp16:
            img_mask = img_mask.half()
        model_kwargs['full_emb1'], model_kwargs['pooled_emb1'] = self.encode_text(text_encoder=self.text_encoder1,
                                                                                  tokenizer=self.tokenizer1,
                                                                                  prompt=prompt,
                                                                                  batch_size=batch_size)
        model_kwargs['full_emb2'], model_kwargs['pooled_emb2'] = self.encode_text(text_encoder=self.text_encoder2,
                                                                                  tokenizer=self.tokenizer2,
                                                                                  prompt=prompt,
                                                                                  batch_size=batch_size)
        if self.task_type == 'inpainting':
            init_img = init_img.to(self.device)
            img_mask = img_mask.to(self.device)
            model_kwargs['inpaint_image'] = (init_img * img_mask)
            model_kwargs['inpaint_mask'] = img_mask

        def model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = self.model(combined, ts, **kwargs)
            eps, rest = model_out[:, :4], model_out[:, 4:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)

        def denoised_fn(x_start, ):
            if denoised_type == 'dynamic_threshold':
                x2 = torch.clone(x_start).cpu().detach().numpy()
                p = dynamic_threshold_v
                s = np.percentile(
                    np.abs(x2), p,
                    axis=tuple(range(1, x2.ndim)))[0]
                s = max(s, 1.0)
                x_start = torch.clip(x_start, -s, s) / s
            elif denoised_type == 'clip_denoised':
                x_start = x_start.clamp(-1, 1)
            return (
                    x_start * (1 - img_mask)
                    + init_img * img_mask
            )

        if self.task_type == 'inpainting':
            denoised_function = denoised_fn
        else:
            denoised_function = None
        self.model.del_cache()
        if sampler == 'p_sampler':
            samples = diffusion.p_sample_loop(
                model_fn,
                (full_batch_size, 4, new_h, new_w),
                device=self.device,
                denoised_type=denoised_type,
                dynamic_threshold_v=dynamic_threshold_v,
                noise=noise,
                progress=progress,
                model_kwargs=model_kwargs,
                init_step=init_step,
                denoised_fn=denoised_function,
            )[:batch_size]
        elif sampler == 'ddim_sampler':
            samples = diffusion.ddim_sample_loop(
                model_fn,
                (full_batch_size, 4, new_h, new_w),
                device=self.device,
                denoised_type=denoised_type,
                dynamic_threshold_v=dynamic_threshold_v,
                noise=noise,
                progress=progress,
                model_kwargs=model_kwargs,
                init_step=init_step,
                denoised_fn=denoised_function,
                eta=ddim_eta
            )[:batch_size]
        else:
            raise ValueError('Only p_sampler and ddim_sampler is available')
        self.model.del_cache()
        if self.use_image_enc:
            if self.use_fp16:
                samples = samples.half()
            samples = self.image_encoder.decode(samples / self.scale)
        samples = samples[:, :, :h, :w]
        return process_images(samples)

    @torch.no_grad()
    def generate_text2img(self, prompt, num_steps=100,
                          batch_size=1, guidance_scale=7, progress=True,
                          dynamic_threshold_v=99.5, denoised_type='dynamic_threshold', h=512, w=512
                          , sampler='ddim_sampler', ddim_eta=0.05):
        config = deepcopy(self.config)
        config['diffusion_config']['timestep_respacing'] = str(num_steps)
        if sampler == 'ddim_sampler':
            config['diffusion_config']['timestep_respacing'] = 'ddim' + config['diffusion_config']['timestep_respacing']
        diffusion = create_gaussian_diffusion(**config['diffusion_config'])
        return self.generate_img(prompt=prompt, batch_size=batch_size,
                                 diffusion=diffusion,
                                 guidance_scale=guidance_scale, progress=progress,
                                 dynamic_threshold_v=dynamic_threshold_v, denoised_type=denoised_type,
                                 h=h, w=w, sampler=sampler, ddim_eta=ddim_eta)

    @torch.no_grad()
    def encode_prompts(self, prompts, batch_size):
        assert isinstance(prompts, (tuple, list)) and len(prompts) == 2
        text_embeds = [{}, {}]

        for i, prompt in enumerate(prompts):
            text_embeds[i]['full_emb1'], text_embeds[i]['pooled_emb1'] = self.encode_text(
                text_encoder=self.text_encoder1,
                tokenizer=self.tokenizer1,
                prompt=prompt,
                batch_size=batch_size)
            text_embeds[i]['full_emb2'], text_embeds[i]['pooled_emb2'] = self.encode_text(
                text_encoder=self.text_encoder2,
                tokenizer=self.tokenizer2,
                prompt=prompt,
                batch_size=batch_size)

        return text_embeds

    @torch.no_grad()
    def generate_img_encoded_prompt(self, model_kwargs, batch_size=1, diffusion=None,
                                    guidance_scale=7, progress=True, dynamic_threshold_v=99.5,
                                    denoised_type='dynamic_threshold', init_step=None, noise=None,
                                    init_img=None, img_mask=None, h=512, w=512, sampler='ddim_sampler', ddim_eta=0.8):
        new_h, new_w = self.get_new_h_w(h, w)
        full_batch_size = batch_size * 2
        if init_img is not None and self.use_fp16:
            init_img = init_img.half()
        if img_mask is not None and self.use_fp16:
            img_mask = img_mask.half()

        if self.task_type == 'inpainting':
            init_img = init_img.to(self.device)
            img_mask = img_mask.to(self.device)
            model_kwargs['inpaint_image'] = (init_img * img_mask)
            model_kwargs['inpaint_mask'] = img_mask

        def model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = self.model(combined, ts, **kwargs)
            eps, rest = model_out[:, :4], model_out[:, 4:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)

        def denoised_fn(x_start, ):
            if denoised_type == 'dynamic_threshold':
                x2 = torch.clone(x_start).cpu().detach().numpy()
                p = dynamic_threshold_v
                s = np.percentile(
                    np.abs(x2), p,
                    axis=tuple(range(1, x2.ndim)))[0]
                s = max(s, 1.0)
                x_start = torch.clip(x_start, -s, s) / s
            elif denoised_type == 'clip_denoised':
                x_start = x_start.clamp(-1, 1)
            return (
                    x_start * (1 - img_mask)
                    + init_img * img_mask
            )

        if self.task_type == 'inpainting':
            denoised_function = denoised_fn
        else:
            denoised_function = None

        self.model.del_cache()
        if sampler == 'p_sampler':
            samples = diffusion.p_sample_loop(
                model_fn,
                (full_batch_size, 4, new_h, new_w),
                device=self.device,
                denoised_type=denoised_type,
                dynamic_threshold_v=dynamic_threshold_v,
                noise=noise,
                progress=progress,
                model_kwargs=model_kwargs,
                init_step=init_step,
                denoised_fn=denoised_function,
            )[:batch_size]
        elif sampler == 'ddim_sampler':
            samples = diffusion.ddim_sample_loop(
                model_fn,
                (full_batch_size, 4, new_h, new_w),
                device=self.device,
                denoised_type=denoised_type,
                dynamic_threshold_v=dynamic_threshold_v,
                noise=noise,
                progress=progress,
                model_kwargs=model_kwargs,
                init_step=init_step,
                denoised_fn=denoised_function,
                eta=ddim_eta
            )[:batch_size]
        else:
            raise ValueError('Only p_sampler and ddim_sampler is available')
        self.model.del_cache()
        if self.use_image_enc:
            if self.use_fp16:
                samples = samples.half()
            samples = self.image_encoder.decode(samples / self.scale)
        samples = samples[:, :, :h, :w]
        return process_images(samples)

    @torch.no_grad()
    def generate_prompt_interpolation(self, prompts, inter_alphas=None,
                     batch_size=1, diffusion=None,
                     guidance_scale=7, progress=True, dynamic_threshold_v=99.5,
                     denoised_type='dynamic_threshold', init_step=None, noise=None,
                     init_img=None, img_mask=None, h=512, w=512, sampler='ddim_sampler', ddim_eta=0.8,
                     ):
        new_h, new_w = self.get_new_h_w(h, w)
        full_batch_size = batch_size * 2
        model_kwargs = {}
        if noise is None:
            noise = torch.randn(full_batch_size, 4, new_h, new_w, device=self.device)
        if self.use_fp16:
            noise = noise.half()

        text_embeds = self.encode_prompts(prompts, batch_size)

        samples_list = []
        if inter_alphas is None:
            inter_alphas = np.linspace(0, 1, 21)

        if progress:
            from tqdm.auto import tqdm
            inter_alphas = tqdm(inter_alphas)

        for alpha in inter_alphas:
            self.model.del_cache()

            for key in text_embeds[0].keys():
                if text_embeds[0][key] is None or text_embeds[1][key] is None:
                    model_kwargs[key] = None
                else:
                    model_kwargs[key] = (1 - alpha) * text_embeds[0][key] + alpha * text_embeds[1][key]

                samples_list += self.generate_img_encoded_prompt(
                    model_kwargs=model_kwargs, batch_size=batch_size, diffusion=diffusion,
                    guidance_scale=guidance_scale, progress=False, dynamic_threshold_v=dynamic_threshold_v,
                    denoised_type=denoised_type, init_step=init_step, noise=noise,
                    init_img=init_img, img_mask=img_mask, new_h=new_h, new_w=new_w,
                    sampler=sampler, ddim_eta=ddim_eta
                )

        return samples_list

    @torch.no_grad()
    def generate_text2img_transition(self, prompts, inter_alphas=None, num_steps=100,
                                     batch_size=1, guidance_scale=7, progress=True,
                                     dynamic_threshold_v=99.5, denoised_type='dynamic_threshold', h=512, w=512,
                                     sampler='ddim_sampler', ddim_eta=0.05):
        config = deepcopy(self.config)
        config['diffusion_config']['timestep_respacing'] = str(num_steps)
        if sampler == 'ddim_sampler':
            config['diffusion_config']['timestep_respacing'] = 'ddim' + config['diffusion_config']['timestep_respacing']
        diffusion = create_gaussian_diffusion(**config['diffusion_config'])
        return self.generate_prompt_interpolation(prompts=prompts, inter_alphas=inter_alphas,
                                                  batch_size=batch_size, diffusion=diffusion,
                                                  guidance_scale=guidance_scale, progress=progress,
                                                  dynamic_threshold_v=dynamic_threshold_v, denoised_type=denoised_type,
                                                  h=h, w=w, sampler=sampler, ddim_eta=ddim_eta)

    @torch.no_grad()
    def generate_img2img(self, prompt, pil_img, strength=0.7,
                         num_steps=100, guidance_scale=7, progress=True,
                         dynamic_threshold_v=99.5, denoised_type='dynamic_threshold'
                         , sampler='ddim_sampler', ddim_eta=0.05):

        config = deepcopy(self.config)
        config['diffusion_config']['timestep_respacing'] = str(num_steps)
        if sampler == 'ddim_sampler':
            config['diffusion_config']['timestep_respacing'] = 'ddim' + config['diffusion_config']['timestep_respacing']
        diffusion = create_gaussian_diffusion(**config['diffusion_config'])
        image = prepare_image(pil_img).to(self.device)
        if self.use_fp16:
            image = image.half()
        image = self.image_encoder.encode(image).sample() * self.scale
        start_step = int(diffusion.num_timesteps * (1 - strength))
        image = q_sample(image, torch.tensor(diffusion.timestep_map[start_step - 1]).to(self.device),
                         schedule_name=config['diffusion_config']['noise_schedule'],
                         num_steps=config['diffusion_config']['steps'])
        image = image.repeat(2, 1, 1, 1)
        return self.generate_img(prompt=prompt, batch_size=1,
                                 diffusion=diffusion, noise=image,
                                 guidance_scale=guidance_scale, progress=progress,
                                 dynamic_threshold_v=dynamic_threshold_v, denoised_type=denoised_type,
                                 init_step=start_step, sampler=sampler, ddim_eta=ddim_eta)

    @torch.no_grad()
    def generate_img2img_transition(self, prompts, pil_imgs, inter_alphas=None,
                                    noise_perturbation=0, prompt_perturbation=0,
                                    strength=0.7, num_steps=100, guidance_scale=7, progress=True,
                                    dynamic_threshold_v=99.5, denoised_type='dynamic_threshold',
                                    sampler='ddim_sampler', ddim_eta=0.05):
        config = deepcopy(self.config)
        config['diffusion_config']['timestep_respacing'] = str(num_steps)
        if sampler == 'ddim_sampler':
            config['diffusion_config']['timestep_respacing'] = 'ddim' + config['diffusion_config']['timestep_respacing']
        diffusion = create_gaussian_diffusion(**config['diffusion_config'])
        start_step = int(diffusion.num_timesteps * (1 - strength))

        def encode_image(pil_img):
            image = prepare_image(pil_img).to(self.device)
            if self.use_fp16:
                image = image.half()
            image = self.image_encoder.encode(image).sample() * self.scale
            image = q_sample(image, torch.tensor(diffusion.timestep_map[start_step - 1]).to(self.device),
                             schedule_name=config['diffusion_config']['noise_schedule'],
                             num_steps=config['diffusion_config']['steps'])
            image = image.repeat(2, 1, 1, 1)
            return image

        model_kwargs = {}
        text_embeds = self.encode_prompts(prompts, 1)
        encoded_images = [encode_image(pil_img) for pil_img in pil_imgs]
        embeds_eps = {}
        if prompt_perturbation > 0:
            for key, embed in text_embeds[0].values():
                embeds_eps[key] = torch.randn_like(embed)

        samples_list = []
        if inter_alphas is None:
            inter_alphas = np.linspace(0, 1, 100)

        if progress:
            from tqdm.auto import tqdm
            inter_alphas = tqdm(inter_alphas)

        for alpha in inter_alphas:
            self.model.del_cache()

            for key in text_embeds[0].keys():
                if text_embeds[0][key] is None or text_embeds[1][key] is None:
                    model_kwargs[key] = None
                else:
                    model_kwargs[key] = (1 - alpha) * text_embeds[0][key] + alpha * text_embeds[1][key]
                    if prompt_perturbation > 0:
                        beta = prompt_perturbation * (1 - 2 * np.abs(alpha - 0.5))
                        model_kwargs[key] = (1 - beta) * model_kwargs[key] + beta * embeds_eps[key]

            noise = np.sqrt(1 - alpha) * encoded_images[0] + np.sqrt(alpha) * encoded_images[1]
            if noise_perturbation > 0:
                eps = torch.randn_like(noise)
                noise = np.sqrt(1 - noise_perturbation) * noise + np.sqrt(noise_perturbation) * eps

            samples_list += self.generate_img_encoded_prompt(
                model_kwargs=model_kwargs, batch_size=1,
                diffusion=diffusion, noise=noise,
                guidance_scale=guidance_scale, progress=False,
                dynamic_threshold_v=dynamic_threshold_v, denoised_type=denoised_type,
                init_step=start_step, sampler=sampler, ddim_eta=ddim_eta
            )

        samples_list += []
        return samples_list

    @torch.no_grad()
    def generate_inpainting(self, prompt, pil_img, img_mask,
                            num_steps=100, guidance_scale=7, progress=True,
                            dynamic_threshold_v=99.5, denoised_type='dynamic_threshold',
                            sampler='ddim_sampler', ddim_eta=0.05):
        config = deepcopy(self.config)
        config['diffusion_config']['timestep_respacing'] = str(num_steps)
        if sampler == 'ddim_sampler':
            config['diffusion_config']['timestep_respacing'] = 'ddim' + config['diffusion_config']['timestep_respacing']
        diffusion = create_gaussian_diffusion(**config['diffusion_config'])
        image = prepare_image(pil_img).to(self.device)
        if self.use_fp16:
            image = image.half()
        image = self.image_encoder.encode(image).sample() * self.scale
        image_shape = tuple(image.shape[-2:])
        img_mask = torch.from_numpy(img_mask).unsqueeze(0).unsqueeze(0)
        img_mask = F.interpolate(
            img_mask, image_shape, mode="nearest",
        )
        img_mask = prepare_mask(img_mask).to(self.device)
        if self.use_fp16:
            img_mask = img_mask.half()
        image = image.repeat(2, 1, 1, 1)
        img_mask = img_mask.repeat(2, 1, 1, 1)
        return self.generate_img(prompt=prompt, batch_size=1,
                                 diffusion=diffusion,
                                 guidance_scale=guidance_scale, progress=progress,
                                 dynamic_threshold_v=dynamic_threshold_v, denoised_type=denoised_type,
                                 init_img=image, img_mask=img_mask, sampler=sampler, ddim_eta=ddim_eta )
