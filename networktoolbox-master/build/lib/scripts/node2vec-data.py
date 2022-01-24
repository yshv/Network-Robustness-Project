import node2vec.src.node2vec
from gensim.models import Word2Vec

def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers,
                     iter=args.iter)
    model.save_word2vec_format(args.output)

    return


def generate_embeddings(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    # READ GRAPH

    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    learn_embeddings(walks)

if __name__ == "__main__":
    main()
