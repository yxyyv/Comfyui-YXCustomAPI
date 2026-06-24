import base64
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from comfy_api.latest import io # type: ignore
from io import BytesIO

import numpy as np
import requests # type: ignore
import torch # type: ignore
from PIL import Image

RESOLUTION_TIMEOUTS = {
    "512px":200,
    "1K": 360,
    "2K": 600,
    "4K": 720,
}

CURRENT_VERSION = "v1.1"
PROVIDER_CONFIG = "provider.json"
MODEL_CONFIG = "configFiles/model.json"
NODE_CATEGORY_1 = "YXCustomAPI"


def _print_image_generation_done(model: str, endpoint_mode: str) -> None:
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"图片生成完成：{model} | {endpoint_mode} | {current_time}")

def _get_config(mode):
    try:
        if mode == "provider":
            config_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), PROVIDER_CONFIG
            )
        elif mode == "model":
            config_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), MODEL_CONFIG
            )
        else:
            print("非法config模式")
            return
        with open(config_path, "r") as f:
            config = json.load(f)
        return config
    except:
        return {}


def _get_enabled_providers():
    """获取配置中启用的 provider 列表"""
    providers = _get_config("provider")
    enabled_providers = []
    for provider_name, provider_config in providers.items():
        if isinstance(provider_config, dict) and provider_config.get("enabled") == 1:
            enabled_providers.append(provider_name)
    return enabled_providers

def _get_model_info(SelectedModel,mode):
    modelInfo = _get_config("model").get("Models",{}).get(SelectedModel)
    if modelInfo is not None:
        if mode == "aspect_ratio":
            output = modelInfo.get("aspect_ratio")
        elif mode == "resolution":
            output = modelInfo.get("resolution")
        elif mode == "aspect_prompt":
            output = modelInfo.get("aspect_prompt")
        elif mode == "display_name":
            output = modelInfo.get("display_name")
        elif mode == "model_name":
            output = modelInfo.get("model_name")
        else:
            print("获取参数类型错误，终止进程")
            return
    else:
        print(f"获取模型信息失败，检查是否包含所需信息，终止进程")
        return

    if output is not None:
        return output
    else:
        print(f"警告：在 {MODEL_CONFIG} 中未找到 '{modelInfo}' 模型配置")
        return []

def _get_model_name_options(model: str) -> List[str]:
    model_names = _get_model_info(model, "model_name")
    if isinstance(model_names, list):
        return [item for item in model_names if item]
    if isinstance(model_names, str) and model_names:
        return [model_names]
    return []


def _parse_ratio(value: str) -> Optional[float]:
    """解析比例字符串为浮点数"""
    try:
        width_str, height_str = value.split(":")
        width = float(width_str)
        height = float(height_str)
        if height == 0:
            return None
        return width / height
    except (ValueError, ZeroDivisionError):
        return None


def _resolve_aspect_ratio(aspect_ratio: str, image: torch.Tensor, model: str) -> str:
    if aspect_ratio != "auto":
        return aspect_ratio
    
    if image is not None:
        # 获取输入图片的实际比例
        image = image[0]
        height = image.shape[1] if image.shape[1] > 0 else 1
        width = image.shape[2] if image.shape[2] > 0 else 1
        actual_ratio = width / height
        
        supported_ratios = _get_model_info(model, "aspect_ratio")
        if supported_ratios is not None:
            supported_ratios = [r for r in supported_ratios if r != "auto"]
        else:
            print("未找到支持的分辨率，默认使用1：1")
            return "1:1"
        
        best_choice = supported_ratios[0]
        best_diff: Optional[float] = None
        
        for candidate in supported_ratios:
            candidate_value = _parse_ratio(candidate)
            if candidate_value is None:
                continue
            diff = abs(actual_ratio - candidate_value)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_choice = candidate
    else:
        return "1:1"
    
    return best_choice


def _load_image_base64(images: torch.Tensor) -> List[Dict[str, Any]]:
    """读取图片并转换为 base64"""
    if images.ndim != 4 or images.shape[-1] != 3:
        raise RuntimeError("images input must be a batch of RGB tensors")

    batch = images.detach().cpu().numpy()
    parts: List[Dict[str, Any]] = []

    for index in range(batch.shape[0]):
        item = batch[index]
        item = np.clip(item, 0.0, 1.0)
        item_uint8 = (item * 255.0).astype(np.uint8)
        image = Image.fromarray(item_uint8, mode="RGB")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        parts.append(
            {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": encoded,
                }
            }
        )

    return parts

def _base64_to_tensor_image(image_data: str) -> torch.Tensor:
    """base64解码为图片"""
    decoded = base64.b64decode(image_data)
    image = Image.open(BytesIO(decoded))
    if image.mode != "RGB":
        image = image.convert("RGB")
    array = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array)
    return tensor.unsqueeze(0)

def _get_requestURL(
    model: str,
    baseURL: str,
    endpointMode: str,
    model_name: str,
) -> str:
    model_config = _get_config("model")
    if not model_config:
        raise RuntimeError(f"Model Config {MODEL_CONFIG} load failed")
    model_info = model_config.get("Models", {}).get(model)
    if not model_info:
        raise RuntimeError(f"Model {model} not found in config")
    request_url_template = model_config.get("Config",{}).get("RequestURL", {}).get(endpointMode)
    if not request_url_template:
        raise RuntimeError(f"RequestURL template not found for endpoint {endpointMode}")

    if endpointMode == "gemini":
        requestURL = (f"{baseURL}{request_url_template}").format(model=model_name)
    elif endpointMode == "gptGenerate" or endpointMode == "gptEdit":
        requestURL = f"{baseURL}{request_url_template}"
    return requestURL

def call_endpoint(
    endpointMode: str,
    api: str,
    prompt: str,
    model: str,
    model_name: str,
    timeout: int,
    resolution: str,
    aspect_ratio: str,
    images: Optional[torch.Tensor]
) -> torch.Tensor:
    """调用 API 端点生成图片，支持 gemini、gptGenerate、gptEdit 模式"""
    api_parts = api.split(",")
    if len(api_parts) >= 2:
        provider = api_parts[0]
        api_key = api_parts[1]
    else:
        raise RuntimeError("Invalid api format, expected 'provider,api_key'")

    chosen_prompt = (prompt or "").strip()
    
    baseURL = _get_config("provider").get(provider,{}).get("baseURL")
    if not baseURL:
        raise RuntimeError(f"baseURL not found for provider {provider}")
    
    model_config = _get_config("model")
    if not model_config:
        raise RuntimeError(f"Model config file {MODEL_CONFIG} load failed")

    model_info = model_config.get("Models", {}).get(model)
    if not model_info:
        raise RuntimeError(f"Model {model} not found in config")

    requestURL = _get_requestURL(model, baseURL,endpointMode,model_name)

    if timeout == -1:
        request_timeout = None
    elif timeout > 0:
        request_timeout = timeout
    else:
        request_timeout = RESOLUTION_TIMEOUTS.get(resolution, 300)

    # 获取比例
    resolved_aspect_ratio = _resolve_aspect_ratio(aspect_ratio, images, model)

    print(f"-------------------------------------------------------------")
    print(f"输入模型：{model}；调用：{model_name}")
    print(f"节点版本：{CURRENT_VERSION}")
    print(f"端点模式：{endpointMode}")
    print(f"当前生图比例：{resolved_aspect_ratio}")
    if images is not None:
        print(f"当前输入图片数量：{len(images)}")
    # print(f"RequestURL:{requestURL}")

    if endpointMode == "gemini":
        return _call_gemini_endpoint(
            requestURL=requestURL,
            api_key=api_key,
            prompt=chosen_prompt,
            images=images,
            aspect_ratio=resolved_aspect_ratio,
            resolution=resolution,
            request_timeout=request_timeout,
            model=model,
        )
    elif endpointMode == "gptGenerate" or endpointMode == "gptEdit":
        return _call_gpt_endpoint(
            requestURL=requestURL,
            api_key=api_key,
            prompt=chosen_prompt,
            aspect_ratio=aspect_ratio,
            request_timeout=request_timeout,
            images=images,
            model=model,
            resolution=resolution,
            endpoint_mode=endpointMode,
        )
    else:
        raise RuntimeError(f"Unknown endpointMode: {endpointMode}")

def _call_gemini_endpoint(
    requestURL: str,
    api_key: str,
    prompt: str,
    images: Optional[torch.Tensor],
    aspect_ratio: str,
    resolution: str,
    request_timeout: int,
    model: str,
) -> torch.Tensor:
    
    if images is not None:
        image0 = images[0]
        if not isinstance(image0, torch.Tensor):
            raise RuntimeError("images input must be a torch tensor")
        if image0.shape[0] == 0:
            raise RuntimeError("at least one image is required")

        parts = _load_image_base64(image0)
        for idx, extra_image in enumerate(images[1:], start=1):
            if extra_image is None:
                continue
            if not isinstance(extra_image, torch.Tensor):
                raise RuntimeError(f"image{idx} input must be a torch tensor")
            if extra_image.shape[0] == 0:
                raise RuntimeError(f"image{idx} input must include at least one image")
            parts.extend(_load_image_base64(extra_image))
        cleaned_prompt = (prompt or "").strip()
        if cleaned_prompt:
            parts.append({"text": cleaned_prompt})
        else:
            print("Warning: empty prompt for gemini image edit request")
    else:
        image0 = None
        parts = [{"text": (prompt or "").strip()}]

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": {
                "aspectRatio": aspect_ratio,
                "imageSize": resolution,
            },
        },
    }

    # 发送请求
    try:
        response = requests.post(
            requestURL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=request_timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"request to {model} API failed: {exc}") from exc

    if response.status_code >= 400:
        raise RuntimeError(f"API状态返回： {response.status_code}: {response.text}")

    try:
        result = response.json()
    except ValueError as exc:
        raise RuntimeError("API Json 解码失败") from exc

    try:
        image_data = result["candidates"][0]["content"]["parts"][0]["inlineData"][
            "data"
        ]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("API回应不包含图像信息") from exc

    result_tensor = _base64_to_tensor_image(image_data)
    _print_image_generation_done(model, "gemini")
    return result_tensor,parts

def _call_gpt_endpoint(
    requestURL: str,
    api_key: str,
    prompt: str,
    aspect_ratio: str,
    request_timeout: int,
    images: Optional[torch.Tensor],
    model: str,
    resolution:str,
    endpoint_mode: str,
)  -> torch.Tensor:
    enhanced_prompt = prompt
    if aspect_ratio == "auto":
        resolved_aspect_ratio = _resolve_aspect_ratio(aspect_ratio,images,model)
        enhanced_prompt = f"{resolved_aspect_ratio},{prompt}"
    else:
        aspect_prompt_map = _get_model_info(model, "aspect_prompt")
        aspect_ratios = _get_model_info(model, "aspect_ratio")
        if aspect_prompt_map and aspect_ratios:
            try:
                idx = aspect_ratios.index(aspect_ratio)
                if idx < len(aspect_prompt_map):
                    enhanced_prompt = f"{aspect_prompt_map[idx]},{prompt}"
            except ValueError:
                pass
    if resolution != "auto":
        enhanced_prompt = f"{resolution},{enhanced_prompt}"
    
    model_name = _get_model_info(model,"model_name")
    if not model_name:
        raise RuntimeError(f"{model}不存在于配置")

    # 图生图
    has_valid_images = images is not None and len(images) > 0 if hasattr(images, '__len__') else images is not None
    if has_valid_images:
        files = []
        for idx, image_tensor in enumerate(images):
            if image_tensor is None:
                continue
            if not isinstance(image_tensor, torch.Tensor):
                raise RuntimeError(f"image{idx} input must be a torch tensor")
            if image_tensor.shape[0] == 0:
                raise RuntimeError(f"image{idx} input must include at least one image")

            if image_tensor.ndim != 4 or image_tensor.shape[-1] != 3:
                raise RuntimeError("images input must be a batch of RGB tensors")

            batch = image_tensor.detach().cpu().numpy()
            for i in range(batch.shape[0]):
                item = batch[i]
                item = np.clip(item, 0.0, 1.0)
                item_uint8 = (item * 255.0).astype(np.uint8)
                img = Image.fromarray(item_uint8, mode="RGB")
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                buffer.seek(0)
                files.append(("image", (f"image_{idx}_{i}.png", buffer, "image/png")))

        if len(files) == 0:
            raise RuntimeError("at least one image is required")

        payload = {
            "model": model_name,
            "prompt": enhanced_prompt,
            "response_format": "b64_json"
        }

    # 文生图
    else:
        files = None
        payload = {
            "model": model_name,
            "prompt": enhanced_prompt,
            "response_format": "b64_json",
        }

    # 发送请求
    try:
        if has_valid_images:
            # 图生图：使用 multipart/form-data
            response = requests.post(
                requestURL,
                headers={"Authorization": f"Bearer {api_key}"},
                data=payload,
                files=files,
                timeout=request_timeout,
            )
        else:
            # 文生图：使用 application/json
            response = requests.post(
                requestURL,
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=request_timeout,
            )
    except requests.RequestException as exc:
        raise RuntimeError(f"request to {model} API failed: {exc}") from exc

    if response.status_code >= 400:
        raise RuntimeError(f"API状态返回： {response.status_code}: {response.text}")

    try:
        result = response.json()
    except ValueError as exc:
        raise RuntimeError("API Json 解码失败") from exc

    try:
        image_data = result["data"][0]["b64_json"]
        if image_data.startswith("data:image/png;base64,"):
            image_data = image_data[len("data:image/png;base64,") :]
        elif image_data.startswith("data:image/jpeg;base64,"):
            image_data = image_data[len("data:image/jpeg;base64,") :]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("API回应不包含图像信息") from exc

    result_tensor = _base64_to_tensor_image(image_data)
    _print_image_generation_done(model, endpoint_mode)
    return result_tensor,enhanced_prompt
    
def create_gemini_node_class(model, node_id, display_name, description, category):
    """工厂函数：动态创建 gemini 节点类"""
    class DynamicGeminiNode(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            resolution_options = _get_model_info(model, "resolution")
            aspect_ratio_options = _get_model_info(model, "aspect_ratio")
            model_name_options = _get_model_name_options(model)
            return io.Schema(
                node_id=node_id,
                display_name=display_name,
                category=category,
                description=description,
                inputs=[
                    io.Custom("Api").Input("api"),
                    io.Image.Input("images",optional=True),
                    io.String.Input("prompt", multiline=True, default=" "),
                    io.Combo.Input("model_name", options=model_name_options, default=model_name_options[0]),
                    io.Combo.Input("resolution", options=resolution_options, default=resolution_options[0]),
                    io.Combo.Input("aspect_ratio", options=aspect_ratio_options, default=aspect_ratio_options[0]),
                    io.Int.Input("timeout", default=0, min=-1, max=1200),
                    io.Int.Input("seed",default=0,min=0,max=99999999),
                ],
                outputs=[
                    io.Image.Output("Generated_Image"),
                    io.String.Output("ContentInput")
                ]
            )

        @classmethod
        def execute(
            cls,
            images=None,
            prompt=None,
            api=None,
            resolution=None,
            aspect_ratio=None,
            timeout=None,
            seed=None,
            model_name=None
        ) -> io.NodeOutput:
            result = call_endpoint(
                endpointMode="gemini",
                api=api,
                prompt=prompt,
                model=model,
                model_name=model_name,
                timeout=timeout,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                images=images,
            )
            result_tensor = result[0]
            result_parts = result[1]
            return io.NodeOutput(result_tensor,result_parts)

    return DynamicGeminiNode

def create_gpt_node_class(model, node_id, display_name, description, category):
    """工厂函数：动态创建 gpt 节点类"""
    class DynamicGPTNode(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            resolution_options = _get_model_info(model, "resolution")
            aspect_ratio_options = _get_model_info(model, "aspect_ratio")
            model_name_options = _get_model_name_options(model)
            return io.Schema(
                node_id=node_id,
                display_name=display_name,
                category=category,
                description=description,
                inputs=[
                    io.Custom("Api").Input("api"),
                    io.Image.Input("images",optional=True),
                    io.String.Input("prompt", multiline=True, default=""),
                    io.Combo.Input("model_name", options=model_name_options, default=model_name_options[0]),
                    io.Combo.Input("resolution", options=resolution_options, default=resolution_options[0]),
                    io.Combo.Input("aspect_ratio", options=aspect_ratio_options, default=aspect_ratio_options[0]),
                    io.Int.Input("timeout", default=0, min=-1, max=1200),
                    io.Int.Input("seed",default=0,min=0,max=99999999),
                ],
                outputs=[
                    io.Image.Output("Generated_Image"),
                    io.String.Output("Prompt")
                ]
            )

        @classmethod
        def execute(
            cls,
            prompt=None,
            api=None,
            images=None,
            model_name=None,
            resolution=None,
            aspect_ratio=None,
            timeout=None,
            seed=None
        ) -> io.NodeOutput:
            has_images = images is not None and len(images) > 0 if hasattr(images, '__len__') else images is not None
            endpoint_mode = "gptEdit" if has_images else "gptGenerate"
            print(f"GPT节点模式：{endpoint_mode} (has_images={has_images})")
            
            result= call_endpoint(
                endpointMode=endpoint_mode,
                api=api,
                prompt=prompt,
                model=model,
                model_name=model_name,
                timeout=timeout,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                images=images,
            )
            result_tensor = result[0]
            result_prompt = result[1]
            return io.NodeOutput(result_tensor,result_prompt)

    return DynamicGPTNode

class YXDynamicImageListNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image"), prefix="image", min=1, max=10
        )
        return io.Schema(
            node_id="YXDynamicImageListNode",
            display_name="Image List",
            category=NODE_CATEGORY_1,
            description="Collect all reference images.",
            inputs=[
                io.Autogrow.Input("images", template=autogrow_template),
            ],
            outputs=[
                io.Image.Output()
            ],
        )

    @classmethod
    def execute(cls, images: io.Autogrow.Type) -> io.NodeOutput:
        raw_image_list = list(images.values())
        return io.NodeOutput(raw_image_list)

class YXAPIInfoNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        enabled_providers = _get_enabled_providers()
        default_provider = enabled_providers[0] if enabled_providers else ""
        return io.Schema(
            node_id="YXAPIInfoNode",
            display_name="API info",
            category=NODE_CATEGORY_1,
            description="Set provider and api infomation.",
            inputs=[
                io.Combo.Input("provider", options=enabled_providers, default=default_provider),
                io.String.Input("api_key",multiline=False, default=""),
            ],
            outputs=[io.Custom("Api").Output(display_name = "api")]
        )

    @classmethod
    def execute(
        cls,
        provider,
        api_key
        ) -> io.NodeOutput:
        provider_info = _get_config("provider").get(provider)
        if api_key == "":
            if provider_info is not None:
                api_key = provider_info.get("api-key")
            else:
                print("找不到模型，终止进程")
                return
        output = f"{provider},{api_key}"
        return io.NodeOutput(output)

YXNanoBanana2APINode = create_gemini_node_class(
    "Nano Banana 2",
    "YXNanoBanana2APINode",
    "Nano Banana 2",
    "Use Nano banana 2 with custom api (gemini-3.1-flash-image-preview)",
    NODE_CATEGORY_1
)

YXNanoBananaProAPINode = create_gemini_node_class(
    "Nano Banana pro",
    "YXNanoBananaProAPINode",
    "Nano Banana Pro",
    "Use Nano banana pro with custom api (gemini-3-pro-image-preview)",
    NODE_CATEGORY_1
)

YXGPTImage2AllAPINode = create_gpt_node_class(
    "GPT-image-2",
    "YXGPTImage2AllAPINode",
    "GPT image 2",
    "Use GPT image 2 all with custom api (gpt-image-2-all)",
    NODE_CATEGORY_1
)    



NODE_CLASS_MAPPINGS = {
    "YXDynamicImageListNode": YXDynamicImageListNode,
    "YXAPIInfoNode": YXAPIInfoNode,
    "YXNanoBanana2APINode": YXNanoBanana2APINode,
    "YXNanoBananaProAPINode": YXNanoBananaProAPINode,
    "YXGPTImage2AllAPINode": YXGPTImage2AllAPINode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YXDynamicImageListNode": "Image List",
    "YXAPIInfoNode": "API info",
    "YXNanoBanana2APINode": "Nano Banana 2",
    "YXNanoBananaProAPINode": "Nano Banana Pro",
    "YXGPTImage2AllAPINode": "GPT image 2"
}
