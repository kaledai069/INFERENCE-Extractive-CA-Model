pip install transformers
pip install faiss-cpu
pip install wordsegment
pip install git+https://github.com/alexdej/puzpy.git
pip install streamlit

mkdir -p /content/Inference_components
wget https://huggingface.co/prajesh069/clue-answer.multi-answer-scoring.dual-bert-encoder/resolve/main/dpr_biencoder_trained_500k.bin -O /content/Inference_components/dpr_biencoder_trained_500k.bin
wget https://huggingface.co/prajesh069/clue-answer.multi-answer-scoring.dual-bert-encoder/resolve/main/all_answer_list.tsv -O /content/Inference_components/all_answer_list.tsv
wget https://huggingface.co/prajesh069/clue-answer.multi-answer-scoring.dual-bert-encoder/resolve/main/embeddings_all_answers_json_0.pkl -O /content/Inference_components/embeddings_all_answers_json_0.pkl