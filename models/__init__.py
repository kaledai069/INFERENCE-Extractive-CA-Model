def init_hf_bert_biencoder(args, **kwargs):
    from .hf_models import get_bert_biencoder_components
    return get_bert_biencoder_components(args, **kwargs)

def init_hf_distilbert_biencoder(args, **kwargs):
    from .hf_models import get_distilbert_biencoder_components
    return get_distilbert_biencoder_components(args, **kwargs)

def init_hf_bert_tenzorizer(args, **kwargs):
    from .hf_models import get_bert_tensorizer
    return get_bert_tensorizer(args)

def init_hf_distilbert_tenzorizer(args, **kwargs):
    from .hf_models import get_distilbert_tensorizer
    return get_distilbert_tensorizer(args)


BIENCODER_INITIALIZERS = {
    'hf_bert': init_hf_bert_biencoder, 
    'hf_distilbert': init_hf_distilbert_biencoder
}

TENSORIZER_INITIALIZERS = {
    'hf_bert': init_hf_bert_tenzorizer, 
    'hf_distilbert': init_hf_distilbert_tenzorizer
}

def init_comp(initializers_dict, type, args, **kwargs):
    if type in initializers_dict:
        return initializers_dict[type](args, **kwargs)
    else:
        raise RuntimeError('unsupported model type: {}'.format(type))

def init_biencoder_components(encoder_type: str, args, **kwargs):
    return init_comp(BIENCODER_INITIALIZERS, encoder_type, args, **kwargs)

def init_tenzorizer(encoder_type: str, args, **kwargs):
    return init_comp(TENSORIZER_INITIALIZERS, encoder_type, args, **kwargs)