import re
from enum import Enum
from typing import Any

from ai_tools import edit_image, generate_image, solve_math, understand_image
from queen import ask_queen, classify_multilingual_intent
from quantum import run_quantum

MAX_HISTORY = 30
MAX_MESSAGE_LENGTH = 20000
QUANTUM_QUBITS = 20
QUANTUM_SHOTS = 1


class Intent(str, Enum):
    NORMAL = "normal"
    MATH = "math"
    QUANTUM = "quantum"
    VISION = "vision"
    IMAGE_GENERATION = "image_generation"
    IMAGE_EDIT = "image_edit"


QUANTUM_PATTERNS = (
    "quantum", "qubit", "qubits", "qiskit", "quantum circuit",
    "quantum computing", "quantum computer", "quantum gate",
    "quantum gates", "superposition", "entanglement", "bell state",
    "quantum algorithm"
)

IMAGE_GENERATION_PATTERNS = (
    # English
    "generate image", "create image", "make an image", "draw an image",
    "generate a picture", "create a picture", "make a picture",
    "generate photo", "create photo", "make photo", "draw", "paint",
    "product image", "product photo", "advertising image", "advertising photo",
    "ad image", "ad photo", "promo image", "promotional image",
    "poster", "thumbnail", "catalog image", "catalog photo", "marketing image",
    "create artwork", "generate artwork",
    # Hindi / Hinglish
    "image bana", "image banao", "photo bana", "photo banao",
    "picture bana", "picture banao", "banao photo",
    "image banana", "image banani", "image banani hai", "image banana hai",
    "photo banana", "photo banani", "photo banani hai", "photo banana hai",
    "image chahiye", "photo chahiye", "picture chahiye", "tasveer chahiye",
    "photo ke liye", "image ke liye", "photo of", "picture of", "image of",
    "ki photo", "ka photo", "ki image", "ka image", "sale ki photo",
    "sale ka photo", "product ki photo", "product ka photo",
    "product ki image", "product ka image", "ad ke liye", "poster bana",
    "thumbnail bana", "catalog photo",
    "इमेज बनाओ", "इमेज बना", "फोटो बनाओ", "फोटो बना",
    "चित्र बनाओ", "चित्र बना", "तस्वीर बनाओ", "तस्वीर बना",
    # Urdu
    "تصویر بناؤ", "تصویر بنا دو", "تصویر بنا", "فوٹو بناؤ", "فوٹو بنا دو",
    "ایک تصویر بناؤ", "ایک تصویر بنا دو", "تصویر تیار کرو",
    # Marathi
    "चित्र बनवा", "चित्र बनव", "फोटो बनवा", "फोटो बनव", "प्रतिमा बनवा",
    "इमेज बनवा", "चित्र तयार करा", "फोटो तयार करा",
    # Spanish
    "crear una imagen", "crea una imagen", "haz una imagen", "hacer una imagen",
    "genera una imagen", "generar una imagen", "crear una foto", "haz una foto",
    # Arabic
    "أنشئ صورة", "انشئ صورة", "اصنع صورة", "صورة أنشئ", "إنشاء صورة",
    "اعمل صورة", "ارسم صورة",
    # French
    "créer une image", "crée une image", "fais une image", "faire une image",
    "génère une image", "générer une image", "créer une photo",
    # Bengali
    "একটি ছবি বানাও", "ছবি বানাও", "ছবি তৈরি কর", "একটি ছবি তৈরি কর",
    "ছবি আঁকো", "ফটো বানাও",
    # Portuguese
    "crie uma imagem", "criar uma imagem", "faça uma imagem", "fazer uma imagem",
    "gere uma imagem", "gerar uma imagem", "crie uma foto", "faça uma foto",
    # Russian
    "создай изображение", "создать изображение", "сделай изображение",
    "сделай картинку", "создай картинку", "создай фото", "нарисуй изображение",
    # Indonesian
    "buat gambar", "buat sebuah gambar", "buat foto", "hasilkan gambar",
    "buatkan gambar", "gambar buat"
)

CAPABILITY_PATTERNS = (
    "can you see images", "can you see image", "can you view images",
    "can you view image", "can you analyze images", "can you analyze image",
    "can you create images", "can you create image", "can you make images",
    "can you make image", "can you generate images", "can you generate image",
    "can you edit images", "can you edit image", "do you support images",
    "do you support image generation", "do you support image editing",
    "what can you do with images", "what can you do with image",
    "क्या तुम इमेज देख", "क्या तुम फोटो देख", "क्या तुम इमेज बना",
    "क्या तुम फोटो बना", "क्या तुम इमेज एडिट", "क्या तुम फोटो एडिट",
    "क्या तुम इमेज कर सकते", "क्या तुम फोटो कर सकते",
)

# Strong edit actions. These are checked only when an image is uploaded,
# so words like "make", "create", or "change" cannot accidentally route
# a normal text-only chat message to FLUX.
IMAGE_EDIT_ACTION_PATTERNS = (
    # English: explicit editing instructions. Avoid single generic verbs such as
    # "change" or "add" because they also occur in normal visual questions.
    "edit this", "edit the image", "edit the photo", "edit the picture",
    "modify this", "modify the image", "modify the photo", "modify the picture",
    "change this", "change the background", "change the color",
    "replace the background", "replace the", "remove the background",
    "remove the", "delete the", "erase the", "add a", "add an",
    "insert a", "insert an", "put a", "put an", "swap the",
    "make this", "make it", "turn this into", "turn it into",
    "create from this", "generate from this", "redraw this",
    "recolor", "resize", "crop", "uncrop", "upscale", "retouch",
    "enhance this", "improve this", "restore this", "clean up this",
    "background change", "background removal",
    # Hinglish / Hindi transliterations
    "bana do", "bana de", "banado", "badal do", "badal de",
    "hata do", "hata de", "nikal do", "nikal de", "jod do", "jod de",
    "laga do", "laga de", "daal do", "dal do",
    "kar do", "kardo", "kar de", "karde", "karna hai", "karna",
    "remove do", "change do", "edit do", "replace do",
    "add kar", "remove kar", "change kar", "edit kar", "modify kar",
    "replace kar", "fix kar",
    "improve kar", "enhance kar", "accha bana", "achha bana",
    "sundar bana", "theek kar", "saaf kar", "bada kar", "chhota kar",
    "upgrade kar", "background change", "background badal",
    # Devanagari
    "बना दो", "बना दे", "बनादो", "बदल दो", "बदल दे",
    "हटा दो", "हटा दे", "निकाल दो", "निकाल दे",
    "जोड़ दो", "जोड़ दे", "डाल दो", "डाल दे", "लगाओ", "लगा दो",
    "एडिट", "बदल", "हटाओ", "जोड़ो", "बढ़ाओ", "घटाओ",
    "सुधारो", "ठीक करो", "अच्छा बना", "अच्छी बना", "सुंदर बना",
    "सुन्दर बना", "साफ करो", "बैकग्राउंड बदल", "बैकग्राउंड हटाओ",
)

# Context phrases that strongly indicate the user is referring to the
# uploaded image and wants a visual modification.
IMAGE_EDIT_CONTEXT_PATTERNS = (
    # English
    "this image", "this photo", "this picture", "in this image",
    "in this photo", "in this picture", "on this image", "on this photo",
    "behind this", "behind it", "in the background", "to the background",
    "foreground", "background", "beside this", "next to this",
    "in front of this", "in front of it", "on top of this",
    "remove the", "add a", "add an", "put a", "put an",
    # Hinglish
    "is image", "is photo", "is picture", "iss image", "iss photo",
    "is wali image", "is wali photo", "is wale photo", "is mein",
    "isme", "iss mein", "iske pichhe", "iske peeche", "is ke pichhe",
    "is ke peeche", "iske saamne", "iske samne", "is ke saamne",
    "is ke samne", "iske upar", "iske neeche", "background mein",
    "background me", "background mai", "peeche", "pichhe", "saamne",
    "samne", "upar", "neeche", "side mein", "side me",
    # Devanagari
    "इस इमेज", "इस फोटो", "इस तस्वीर", "इस चित्र", "इसमें",
    "इस में", "इसके पीछे", "इसके पिच्छे", "इस के पीछे", "इसके सामने",
    "इस के सामने", "इसके ऊपर", "इस के ऊपर", "इसके नीचे", "इस के नीचे",
    "बैकग्राउंड में", "पीछे", "सामने", "ऊपर", "नीचे", "साइड में",
)

# Kept for compatibility/readability: these are the original explicit
# edit phrases plus the broader patterns above.
IMAGE_EDIT_PATTERNS = (
    "edit image", "edit this image", "modify image", "modify this image",
    "change image", "change this image", "transform image", "transform this image",
    "background change", "change the background", "remove background",
    "replace background", "change color", "make it", "turn it into",
    "make this photo", "make this image", "improve this photo", "enhance this image",
    "फोटो बदल", "इमेज बदल", "बैकग्राउंड बदल", "background बदल",
    "edit करो", "फोटो को बदल", "इमेज को बदल", "फोटो में बदल",
    "इमेज में बदल", "रंग बदल", "बैकग्राउंड हटा",
    "फोटो को", "इमेज को", "तस्वीर को", "चित्र को",
    "फोटो में", "इमेज में", "तस्वीर में", "चित्र में",
    "अच्छा बना", "अच्छी बना", "सुंदर बना", "सुन्दर बना",
    "बदल दो", "बना दो", "कर दो", "हटा दो", "जोड़ दो",
    "बैकग्राउंड हटाओ", "बैकग्राउंड बदलो", "रंग बदलो",
    "फोटो एडिट", "इमेज एडिट", "तस्वीर एडिट",
)

MATH_PATTERNS = (
    "solve equation", "solve this equation", "calculate", "mathematics",
    "mathematical", "math problem", "math question", "algebra",
    "geometry", "trigonometry", "percentage", "fraction", "quadratic",
    "derivative", "integral", "probability", "गणित", "मैथ",
    "गणित का सवाल", "सवाल हल", "समीकरण", "प्रतिशत"
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def contains_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    normalized = normalize(text)
    return any(p in normalized for p in patterns)


def clean_history(history: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    if not history:
        return []

    return [
        {"role": item["role"], "content": item["content"].strip()[:MAX_MESSAGE_LENGTH]}
        for item in history[-MAX_HISTORY:]
        if isinstance(item, dict)
        and item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
        and item["content"].strip()
    ]


def looks_like_math_structure(message: str) -> bool:
    text = normalize(message)
    equation = bool(re.search(r"\b[a-z]\s*[\+\-\*/^]\s*[\dx0-9]", text))
    equality = "=" in text
    arithmetic = bool(re.search(r"\d+\s*[\+\-\*/]\s*\d+", text))
    fraction = bool(re.search(r"\d+\s*/\s*\d+", text))
    return equation or equality or arithmetic or fraction


def looks_like_image_edit(message: str) -> bool:
    """
    Detect an actual edit command without stealing ordinary vision questions.
    Examples that must stay VISION:
      "What is in the background?"
      "Who is this?"
      "What color is the shirt?"
    """
    text = normalize(message)

    # Natural-language inspection questions should go to vision unless the
    # user also gave an explicit edit command.
    inspection_starts = (
        "what ", "what's ", "who ", "which ", "where ", "when ", "why ",
        "how many", "how much", "describe ", "analyze ", "tell me ",
        "can you see ", "do you see ", "is there ", "are there ",
        "क्या ", "कौन ", "क्या है", "इसमें क्या", "पीछे क्या",
        "کتنا", "کون", "کیا",
    )
    if text.endswith("?") or any(text.startswith(prefix) for prefix in inspection_starts):
        explicit_question_edits = (
            "edit this", "edit the image", "edit the photo", "edit the picture",
            "remove the background", "remove the", "add a", "add an",
            "change the background", "change the color", "replace the",
            "bana do", "badal do", "hata do", "nikal do", "jod do",
            "kar do", "kardo", "kar de", "karde",
            "निकाल दो", "हटा दो", "बदल दो", "जोड़ दो", "बना दो",
            "फोटो एडिट", "इमेज एडिट", "तस्वीर एडिट",
        )
        return contains_pattern(text, explicit_question_edits)

    # Common Hinglish edit commands such as:
    # "iska background black kar do"
    # "is photo ko clean kar do"
    # "background hata do"
    # These are not inspection questions; they explicitly request a change.
    edit_verbs = (
        "kar do", "kardo", "kar de", "karde", "karna hai",
        "bana do", "bana de", "badal do", "badal de",
        "hata do", "hata de", "nikal do", "nikal de",
        "jod do", "jod de", "laga do", "laga de",
        "change kar", "edit kar", "remove kar", "add kar",
        "replace kar", "modify kar", "fix kar",
    )
    edit_context = (
        "background", "colour", "color", "remove", "replace", "add",
        "delete", "erase", "change", "edit", "modify", "crop",
        "resize", "upscale", "retouch", "enhance", "clean",
        "improve", "black", "white",
    )
    if any(v in text for v in edit_verbs) and any(c in text for c in edit_context):
        return True

    return contains_pattern(text, IMAGE_EDIT_ACTION_PATTERNS) or contains_pattern(
        text, IMAGE_EDIT_PATTERNS
    )


def looks_like_image_generation(message: str) -> bool:
    text = normalize(message)
    if not text or contains_pattern(text, CAPABILITY_PATTERNS):
        return False

    visual_words = (
        "image", "images", "photo", "photos", "picture", "pictures",
        "चित्र", "इमेज", "फोटो", "तस्वीर",
        "تصویر", "فوٹو", "صورة",
        "imagen", "foto", "imagem", "fotografia",
        "изображение", "картинка", "фото",
        "ছবি", "ফটো", "gambar",
    )
    request_signals = (
        "generate", "create", "make", "draw", "paint", "design",
        "want an", "need an", "need a", "show me", "send me",
        "give me", "chahiye", "chahiye hai", "banana", "banani",
        "bana", "banao", "banado", "banani hai", "banana hai",
        "दिखाओ", "चाहिए", "बनाना", "बनानी", "बनाओ", "बना दो",
        "تصویر بناؤ", "تصویر بنا", "انشئ", "اصنع",
        "crear", "genera", "creer", "crée", "fais",
        "crie", "gere", "gerar", "создай", "сделай", "нарисуй",
        "buat", "buatkan", "hasilkan",
        "की फोटो", "का फोटो", "की इमेज", "का इमेज",
        "के लिए फोटो", "के लिए इमेज", "product", "sale", "poster",
        "thumbnail", "catalog", "advertising", "advertisement", "promo",
    )
    if not any(word in text for word in visual_words):
        return False
    if any(signal in text for signal in request_signals):
        return True

    # Phrases such as "photo of Taj Mahal" or "Patanjali ki photo" are
    # naturally image requests even when "generate" is omitted.
    return bool(re.search(
        r"(?:image|photo|picture|फोटो|इमेज|तस्वीर|चित्र|تصویر|فوٹو|صورة)"
        r"\s+(?:of|for|की|का|के|کے|لئے|लिए)\s+\S+",
        text,
    ))


def classify_request(
    message: str,
    has_image: bool = False,
) -> Intent:
    # Capability questions must never accidentally trigger generation just because
    # they contain phrases such as "can you create image?".
    if contains_pattern(message, CAPABILITY_PATTERNS):
        return Intent.NORMAL

    # An uploaded image changes the meaning of common generation/edit words.
    # If the user gives an image plus an edit-style instruction, ALWAYS use the
    # image editor instead of sending the request to the vision/text model.
    if has_image:
        if looks_like_image_edit(message):
            return Intent.IMAGE_EDIT
        semantic_intent = classify_multilingual_intent(message, has_image=True)
        if semantic_intent == "IMAGE_EDIT":
            return Intent.IMAGE_EDIT
        return Intent.VISION

    if contains_pattern(message, IMAGE_GENERATION_PATTERNS) or looks_like_image_generation(message):
        return Intent.IMAGE_GENERATION

    # Handle natural Hindi/Hinglish phrasing such as:
    # "ye wali image banao", "is photo ko bana do", "image bana do".
    # Capability questions were already excluded above.
    image_words = (
        "image", "images", "photo", "picture", "इमेज", "फोटो", "तस्वीर", "चित्र",
        "تصویر", "فوٹو", "تصوير", "صورة", "صويرة",
        "imagen", "foto", "imagen", "image", "photo",
        "ছবি", "ফটো", "imagem", "fotografia",
        "изображение", "картинка", "фото",
        "gambar", "foto", "चित्र", "प्रतिमा"
    )
    create_words = (
        "bana", "banao", "banado", "bana do", "बना", "बनाओ", "बना दो",
        "create", "generate", "draw", "make", "paint",
        "بناؤ", "بنا دو", "بنائیں", "بناؤ", "بنव", "बनवा",
        "crear", "crea", "haz", "hacer", "genera", "generar",
        "أنشئ", "اصنع", "إنشاء", "اعمل", "ارسم",
        "créer", "crée", "fais", "faire", "génère", "générer",
        "বানাও", "তৈরি কর", "আঁকো",
        "crie", "criar", "faça", "fazer", "gere", "gerar",
        "создай", "создать", "сделай", "нарисуй",
        "buat", "buatkan", "hasilkan",
        "बनवा", "बनवा", "तयार करा"
    )
    normalized_message = normalize(message)
    if (
        any(word in normalized_message for word in image_words)
        and any(word in normalized_message for word in create_words)
    ) or looks_like_image_generation(message):
        return Intent.IMAGE_GENERATION

    if contains_pattern(message, QUANTUM_PATTERNS):
        return Intent.QUANTUM
    if contains_pattern(message, MATH_PATTERNS) or looks_like_math_structure(message):
        return Intent.MATH

    semantic_intent = classify_multilingual_intent(message, has_image=False)
    try:
        return Intent(semantic_intent.lower())
    except ValueError:
        return Intent.NORMAL


def process_request(
    user_message: str,
    history: list[dict[str, Any]] | None = None,
    image_data: str | None = None,
    image_mime: str | None = None,
    image_url: str | None = None,
) -> dict[str, Any]:
    message = user_message.strip()

    # Some frontends upload the image first and send its public URL rather
    # than a base64 payload. Treat that URL as the actual image input.
    if not image_data and image_url:
        image_data = image_url
    if not message:
        raise ValueError("Please enter a message.")

    cleaned_history = clean_history(history)
    intent = classify_request(message, has_image=bool(image_data))

    if intent == Intent.MATH:
        res = solve_math(message)
        if res.get("solved"):
            return {"type": "math", "status": "completed", "answer": res["answer"], "math": res}
        answer = ask_queen(message, history=cleaned_history, system_context=res.get("context", ""))
        return {"type": "math", "status": "completed", "answer": answer, "math": res}

    if intent == Intent.QUANTUM:
        q_res = run_quantum(num_qubits=QUANTUM_QUBITS, shots=QUANTUM_SHOTS)
        prompt = f"User question: {message}\nQuantum computation: {q_res}"
        answer = ask_queen(prompt, history=cleaned_history)
        return {"type": "quantum", "status": "completed", "answer": answer, "quantum": q_res}

    if intent == Intent.IMAGE_EDIT:
        img_res = edit_image(
            prompt=message,
            image_data=image_data,
            image_mime=image_mime,
        )
        edit_status = img_res.get("status", "completed")
        edit_data = img_res.get("data")
        if edit_status == "completed" and not (
            isinstance(edit_data, str)
            and edit_data.startswith("data:image/")
        ):
            img_res = {
                **img_res,
                "status": "error",
                "answer": "Image editing did not return the edited image.",
                "error": "The image editor returned no data:image result.",
            }

        return {
            "type": "image_edit",
            "status": img_res.get("status", "completed"),
            "answer": img_res.get("answer", ""),
            "image": img_res,
        }

    if intent == Intent.VISION:
        v_res = understand_image(question=message, image_data=image_data, image_mime=image_mime)
        return {"type": "vision", "status": v_res.get("status", "completed"), "answer": v_res.get("answer", ""), "vision": v_res}

    if intent == Intent.IMAGE_GENERATION:
        img_res = generate_image(message)
        return {"type": "image", "status": img_res.get("status", "completed"), "answer": img_res.get("answer", ""), "image": img_res}

    capability_context = (
        "The application has integrated image capabilities. It can analyze uploaded images, "
        "generate new images, and edit existing images using specialized tools. "
        "Do not claim that Quantum Queen AI lacks these capabilities. "
        "If the user is asking what the application can do, describe the application capabilities "
        "accurately rather than pretending the chat model itself must perform every capability."
        if contains_pattern(message, CAPABILITY_PATTERNS)
        else None
    )
    answer = ask_queen(message, history=cleaned_history, system_context=capability_context)
    return {"type": "normal", "status": "completed", "answer": answer}
