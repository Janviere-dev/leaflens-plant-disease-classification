
"""
Farmer-friendly explanations and recommendations for LeafLens.

Guidance is educational, not a confirmed diagnosis or a substitute
for locally appropriate agricultural advice.
"""

import re


HEALTHY_GUIDANCE = {
    "explanation": (
        "The model did not identify a known disease pattern in this image. "
        "The leaf appears consistent with the healthy class."
    ),
    "actions": [
        "Continue providing appropriate water and nutrients.",
        "Maintain suitable sunlight and growing conditions.",
        "Inspect leaves regularly for unusual spots, discoloration or damage.",
        "Maintain good field hygiene and monitor surrounding plants.",
    ],
    "priority": "Routine monitoring",
}


# Keys are normalized disease names. Where the same disease affects
# multiple crops, the guidance is shared unless a crop-specific entry exists.
DISEASE_GUIDANCE = {
    "apple scab": {
        "explanation": (
            "Apple scab is a fungal disease that can cause dark, scabby "
            "spots on leaves and fruit."
        ),
        "actions": [
            "Inspect leaves and fruit for dark, scabby spots.",
            "Remove fallen infected leaves where practical.",
            "Maintain good airflow through appropriate orchard management.",
            "Ask a local agricultural specialist about suitable management.",
        ],
        "priority": "Monitor and seek agricultural advice",
    },
    "black rot": {
        "explanation": (
            "Black rot is a fungal disease that can affect leaves, fruit "
            "and other parts of a plant."
        ),
        "actions": [
            "Inspect leaves and fruit for dark lesions or signs of decay.",
            "Remove affected plant material where practical.",
            "Avoid leaving infected plant debris around the crop.",
            "Seek local advice about appropriate disease management.",
        ],
        "priority": "Monitor and seek agricultural advice",
    },
    "cedar apple rust": {
        "explanation": (
            "Cedar apple rust is a fungal disease that can produce "
            "yellow-orange spots on apple leaves."
        ),
        "actions": [
            "Inspect leaves for orange or yellow spots.",
            "Monitor other apple trees for similar symptoms.",
            "Seek local agricultural advice if symptoms are widespread.",
        ],
        "priority": "Monitor and seek agricultural advice",
    },
    "powdery mildew": {
        "explanation": (
            "Powdery mildew is a fungal disease that commonly produces "
            "a white or powdery coating on leaves."
        ),
        "actions": [
            "Inspect leaves for white powdery growth.",
            "Improve airflow around plants where practical.",
            "Avoid unnecessary overhead watering.",
            "Ask a local specialist about suitable disease management.",
        ],
        "priority": "Monitor and seek agricultural advice",
    },
    "cercospora leaf spot gray leaf spot": {
        "explanation": (
            "Cercospora leaf spot, also called gray leaf spot, is a fungal "
            "disease that produces lesions on corn leaves."
        ),
        "actions": [
            "Inspect leaves for rectangular gray or tan lesions.",
            "Monitor nearby corn plants for similar symptoms.",
            "Manage crop residues using locally recommended practices.",
            "Seek advice if symptoms are spreading.",
        ],
        "priority": "Monitor and seek agricultural advice",
    },
    "common rust": {
        "explanation": (
            "Common rust is a fungal disease that can produce reddish-brown "
            "rust-like pustules on corn leaves."
        ),
        "actions": [
            "Inspect leaves for reddish-brown raised pustules.",
            "Monitor the spread of symptoms across the field.",
            "Maintain appropriate crop management practices.",
            "Seek advice if the disease appears widespread.",
        ],
        "priority": "Monitor and seek agricultural advice",
    },
    "northern leaf blight": {
        "explanation": (
            "Northern corn leaf blight is a fungal disease associated "
            "with long, cigar-shaped lesions on corn leaves."
        ),
        "actions": [
            "Inspect leaves for long, elongated lesions.",
            "Monitor nearby plants for similar symptoms.",
            "Manage infected crop residues appropriately.",
            "Consult a local agricultural specialist.",
        ],
        "priority": "Monitor and seek agricultural advice",
    },
    "esca black measles": {
        "explanation": (
            "Esca, also known as black measles, is a complex grapevine "
            "disease that can affect leaves and woody parts of the vine."
        ),
        "actions": [
            "Inspect vines for unusual leaf discoloration.",
            "Monitor affected vines and note symptom progression.",
            "Avoid moving potentially affected vine material.",
            "Seek advice from a grape production specialist.",
        ],
        "priority": "Seek agricultural advice",
    },
    "leaf blight isariopsis leaf spot": {
        "explanation": (
            "Grape leaf blight, also called Isariopsis leaf spot, can "
            "cause spots and damaged areas on grape leaves."
        ),
        "actions": [
            "Inspect leaves for spots and expanding damaged areas.",
            "Remove affected fallen leaves where practical.",
            "Maintain good airflow around vines.",
            "Seek local advice if symptoms continue spreading.",
        ],
        "priority": "Monitor and seek agricultural advice",
    },
    "haunglongbing citrus greening": {
        "explanation": (
            "Huanglongbing, also known as citrus greening, is a serious "
            "citrus disease that can affect leaves, fruit and tree health."
        ),
        "actions": [
            "Contact a local agricultural extension officer or plant-health specialist.",
            "Have the tree inspected to confirm the suspected disease.",
            "Avoid moving potentially affected citrus plant material.",
            "Ask about locally recommended measures to limit disease spread.",
        ],
        "priority": "Seek specialist advice promptly",
    },
    "bacterial spot": {
        "explanation": (
            "Bacterial spot can cause dark, water-soaked or angular "
            "lesions on leaves and may affect other plant parts."
        ),
        "actions": [
            "Inspect leaves for dark or water-soaked spots.",
            "Avoid handling wet plants unnecessarily.",
            "Remove severely affected plant material where practical.",
            "Seek local advice about appropriate disease management.",
        ],
        "priority": "Monitor and seek agricultural advice",
    },
    "early blight": {
        "explanation": (
            "Early blight is a fungal disease that can produce brown leaf spots, "
            "often with concentric ring patterns."
        ),
        "actions": [
            "Inspect leaves for brown spots with ring-like patterns.",
            "Remove severely affected leaves where practical.",
            "Avoid prolonged leaf wetness where possible.",
            "Seek advice about locally appropriate disease management.",
        ],
        "priority": "Monitor and seek agricultural advice",
    },
    "late blight": {
        "explanation": (
            "Late blight is a potentially destructive disease that can "
            "spread rapidly in suitable cool and humid conditions."
        ),
        "actions": [
            "Inspect leaves for expanding dark or water-soaked lesions.",
            "Seek agricultural advice promptly to confirm the suspected disease.",
            "Avoid moving potentially infected plant material.",
            "Follow locally recommended disease-control measures.",
        ],
        "priority": "Seek specialist advice promptly",
    },
    "leaf scorch": {
        "explanation": (
            "Strawberry leaf scorch can cause reddish-purple spots "
            "and scorched or dried areas on leaves."
        ),
        "actions": [
            "Inspect leaves for reddish-purple spots and dry areas.",
            "Remove affected fallen leaves where practical.",
            "Maintain appropriate watering and growing conditions.",
            "Seek advice if symptoms continue spreading.",
        ],
        "priority": "Monitor and seek agricultural advice",
    },
    "leaf mold": {
        "explanation": (
            "Tomato leaf mold is a fungal disease often associated with "
            "yellow patches on the upper leaf surface and mold growth underneath."
        ),
        "actions": [
            "Inspect both upper and lower leaf surfaces.",
            "Improve ventilation around tomato plants.",
            "Avoid unnecessary overhead watering.",
            "Seek locally appropriate disease-management advice.",
        ],
        "priority": "Monitor and seek agricultural advice",
    },
    "septoria leaf spot": {
        "explanation": (
            "Septoria leaf spot is a fungal disease that produces small "
            "circular spots, often with pale centers and dark edges."
        ),
        "actions": [
            "Inspect leaves for small circular spots.",
            "Remove affected fallen leaves where practical.",
            "Avoid splashing soil onto leaves during watering.",
            "Seek local advice if symptoms spread.",
        ],
        "priority": "Monitor and seek agricultural advice",
    },
    "spider mites two spotted spider mite": {
        "explanation": (
            "Two-spotted spider mites are tiny pests that feed on plant "
            "cells and can cause fine stippling, discoloration and webbing."
        ),
        "actions": [
            "Inspect the undersides of leaves for mites and fine webbing.",
            "Monitor plants for increasing leaf discoloration.",
            "Seek advice about appropriate integrated pest management.",
            "Avoid applying pesticides without confirming the pest.",
        ],
        "priority": "Inspect and seek pest-management advice",
    },
    "target spot": {
        "explanation": (
            "Target spot is a disease that can cause circular leaf lesions "
            "with concentric patterns."
        ),
        "actions": [
            "Inspect leaves for circular lesions.",
            "Improve airflow around plants where practical.",
            "Remove severely affected fallen leaves where appropriate.",
            "Seek local disease-management advice.",
        ],
        "priority": "Monitor and seek agricultural advice",
    },
    "tomato yellow leaf curl virus": {
        "explanation": (
            "Tomato yellow leaf curl virus is associated with leaf curling, "
            "yellowing and reduced plant growth."
        ),
        "actions": [
            "Seek agricultural advice to confirm the suspected virus.",
            "Inspect plants for whiteflies, which can spread the virus.",
            "Monitor nearby tomato plants for similar symptoms.",
            "Ask about locally recommended vector-management practices.",
        ],
        "priority": "Seek specialist advice promptly",
    },
    "tomato mosaic virus": {
        "explanation": (
            "Tomato mosaic virus symptoms can include irregular light and "
            "dark green patterns, leaf distortion and reduced plant growth."
        ),
        "actions": [
            "Seek specialist advice to confirm the suspected viral disease.",
            "Avoid unnecessary handling when moving between plants.",
            "Clean tools using appropriate hygiene practices.",
            "Ask about locally recommended measures to limit spread.",
        ],
        "priority": "Seek agricultural advice promptly",
    },
}


# Apple and grape black rot share a disease name but need crop-aware wording.
CROP_SPECIFIC_GUIDANCE = {
    ("apple", "black rot"): {
        "explanation": (
            "Apple black rot is a fungal disease that can cause leaf spots "
            "and fruit decay."
        ),
        "actions": [
            "Inspect leaves, branches and fruit for dark lesions or decay.",
            "Remove affected plant material where practical.",
            "Avoid leaving infected fruit or plant debris in the orchard.",
            "Seek local advice about appropriate disease management.",
        ],
        "priority": "Monitor and seek agricultural advice",
    },
    ("grape", "black rot"): {
        "explanation": (
            "Grape black rot is a fungal disease that can cause leaf spots "
            "and damage to grape berries."
        ),
        "actions": [
            "Inspect leaves and grape clusters for dark spots or shriveled berries.",
            "Remove affected plant debris and berries where practical.",
            "Maintain good airflow through appropriate vine management.",
            "Seek local agricultural advice.",
        ],
        "priority": "Monitor and seek agricultural advice",
    },
}


def _normalize(value: str) -> str:
    """Normalize labels so punctuation and underscores do not affect matching."""
    value = value.replace("_", " ").replace(",", " ")
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", value.lower()).split())


def get_disease_guidance(crop: str, condition: str) -> dict:
    """
    Return farmer-friendly guidance for a predicted crop condition.
    Matches the crop and condition returned by the model.
    """
    normalized_crop = _normalize(crop)
    normalized_condition = _normalize(condition)

    if normalized_condition == "healthy":
        return {
            **HEALTHY_GUIDANCE,
            "crop": crop,
            "condition": condition,
            "is_healthy": True,
        }

    guidance = CROP_SPECIFIC_GUIDANCE.get(
        (normalized_crop, normalized_condition)
    )

    if guidance is None:
        guidance = DISEASE_GUIDANCE.get(normalized_condition)

    if guidance is not None:
        return {
            **guidance,
            "crop": crop,
            "condition": condition,
            "is_healthy": False,
        }

    # Safe fallback for any unexpected or future model label.
    return {
        "crop": crop,
        "condition": condition,
        "is_healthy": False,
        "explanation": (
            f"The model classified this image as {condition}. "
            "Specific guidance is not yet available for this condition, "
            "so confirmation by an agricultural specialist is recommended."
        ),
        "actions": [
            "Inspect the plant carefully for visible symptoms.",
            "Monitor nearby plants for similar changes.",
            "Consult a local agricultural extension officer or specialist.",
            "Avoid applying treatments until the condition is confirmed.",
        ],
        "priority": "Seek agricultural advice",
    }
