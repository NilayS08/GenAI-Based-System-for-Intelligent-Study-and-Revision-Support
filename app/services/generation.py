import google.generativeai as genai
from app.config import settings

genai.configure(api_key=settings.gemini_api_key)

model = genai.GenerativeModel(model_name=settings.llm_model)


def generate_content(prompt: str) -> str:
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": settings.llm_temperature,
            "max_output_tokens": settings.llm_max_tokens,
        },
    )

    if response and hasattr(response, "text"):
        return response.text

    return "No response generated."
