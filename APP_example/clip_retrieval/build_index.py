import pickle
import faiss
import numpy as np
import click

@click.command()
@click.option('--embeddings_path', default='results/embeddings.pkl', help='Path to the embeddings pickle file.')
@click.option('--save_path', default='results/index.faiss', help='Path to save the faiss index.')
def create_faiss_index(embeddings_path, save_path):
    with open(embeddings_path, 'rb') as f:
        results = pickle.load(f)

    embeddings = np.array(results['embedding'], dtype=np.float32)

    index = faiss.index_factory(embeddings.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
    index.add(embeddings)
    # save index
    faiss.write_index(index, save_path)

if __name__ == "__main__":
    create_faiss_index()
