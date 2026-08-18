import torch
import torch.nn.functional as F

from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel


class SubjectMetrics:
    """
    Compute evaluation metrics which are related to the single subject.
    Those metrics are CLIP-I, CLIP-T, DINO-I.
    """

    def __init__(self, clip_model_id='openai/clip-vit-base-patch32', dino_model_id='facebook/dinov2-base', device='cuda'):
        self.device = device

        # Loading CLIP evaluator
        self.model = CLIPModel.from_pretrained(
            clip_model_id,
            use_safetensors=True
        ).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_id)

        # Loading DINO evaluator
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_model_id)
        self.dino_model = AutoModel.from_pretrained(
            dino_model_id,
            use_safetensors=True
        ).to(self.device)

    def compute_clip_t(self, generated_image: Image.Image, prompt: str) -> float:
        inputs = self.processor(text=[prompt], images=generated_image, return_tensors='pt', padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

        similarity = F.cosine_similarity(image_embeds, text_embeds)
        return similarity.item()

    def compute_clip_i(self, generated_image: Image.Image, reference_images: list[Image.Image]) -> float:
        gen_inputs = self.processor(images=generated_image, return_tensors='pt').to(self.device)
        ref_inputs = self.processor(images=reference_images, return_tensors='pt').to(self.device)

        with torch.no_grad():
            gen_embeds = self.model.get_image_features(**gen_inputs)
            ref_embeds = self.model.get_image_features(**ref_inputs)

            if not isinstance(gen_embeds, torch.Tensor):
                gen_embeds = getattr(gen_embeds, 'image_embeds', gen_embeds[0])
            if not isinstance(ref_embeds, torch.Tensor):
                ref_embeds = getattr(ref_embeds, 'image_embeds', ref_embeds[0])

        similarities = F.cosine_similarity(gen_embeds.unsqueeze(1), ref_embeds.unsqueeze(0), dim=-1)
        return similarities.mean().item()

    def compute_dino_i(self, generated_image: Image.Image, reference_images: list[Image.Image]) -> float:
        gen_inputs = self.dino_processor(images=generated_image, return_tensors='pt').to(self.device)
        ref_inputs = self.dino_processor(images=reference_images, return_tensors='pt').to(self.device)

        with torch.no_grad():
            gen_outputs = self.dino_model(**gen_inputs)
            ref_outputs = self.dino_model(**ref_inputs)

            gen_embeds = F.normalize(gen_outputs.last_hidden_state[:, 0], dim=-1)
            ref_embeds = F.normalize(ref_outputs.last_hidden_state[:, 0], dim=-1)

        similarities = F.cosine_similarity(gen_embeds.unsqueeze(1), ref_embeds.unsqueeze(0), dim=-1)
        return similarities.mean().item()