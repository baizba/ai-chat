import structlog
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

logger = structlog.getLogger()

MODELS = {
    "BAAI/bge-reranker-large": "reranker",
    "microsoft/Phi-3-mini-4k-instruct": "llm",
}


# call this once (possibly from Docker file to cache the models in the beginning
def init_models() -> None:
    for model_name, model_type in MODELS.items():
        init_model(model_name, model_type)


def init_model(model_name: str, model_type: str) -> None:
    logger.info("model.init.start", model_name=model_name, model_type=model_type)

    try:
        AutoTokenizer.from_pretrained(model_name)

        if model_type == "reranker":
            AutoModelForSequenceClassification.from_pretrained(model_name)
        elif model_type == "llm":
            AutoModelForCausalLM.from_pretrained(model_name)
        else:
            raise RuntimeError(f"Unknown model type: {model_type} for {model_name}")

    except Exception as e:
        logger.error("model.init.failed", model_name=model_name, model_type=model_type, error=str(e))
        raise

    logger.info("model.init.end", model_name=model_name, model_type=model_type)
