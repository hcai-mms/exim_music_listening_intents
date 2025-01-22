import os.path

from sentence_transformers import SentenceTransformer, util, CrossEncoder
import networkx as nx

import pandas as pd
from tqdm import tqdm

import pickle

tqdm.pandas()


def add_text_to_question_ensemble(df_fun, models, G, used_text, topK, idx_to_check, factor, used_embeddings,
                                  model_name):
    if topK > 0:

        for i, q in enumerate(list(df_fun[used_text])):
            q_others = [q_other for i_other, q_other in enumerate(df_fun["question"]) if i != i_other]
            q_idxs = [i_other for i_other, q_other in enumerate(df_fun["question"]) if i != i_other]

            mean_scores = None

            for model in models:
                embeddings = model.encode([q] + q_others, normalize_embeddings=True)
                scores = util.dot_score(embeddings[0], embeddings[1:])[0]

                if mean_scores is None:
                    mean_scores = scores
                else:
                    mean_scores += scores

            mean_scores /= len(models)
            idxs = (-mean_scores).argsort()

            count = 0
            last_score = 0.0
            for idx in idxs[:topK]:

                q_idx = q_idxs[idx]
                q_2 = q_others[idx]

                if i not in G:
                    G.add_node(i, text=q)
                if q_idx not in G:
                    G.add_node(q_idx, text=q_2)

                if not G.has_edge(i, q_idx):
                    G.add_edge(i, q_idx, weight=factor * mean_scores[idx])
                else:
                    G[i][q_idx]['weight'] += factor * mean_scores[idx]


def add_text_ensemble(df_fun, models, G, used_text, topK, idx_to_check, factor, used_embeddings, model_name):
    if topK > 0:
        for i, q in enumerate(list(df_fun[used_text])):
            q_others = [q_other for i_other, q_other in enumerate(df_fun[used_text]) if i != i_other]
            q_idxs = [i_other for i_other, q_other in enumerate(df_fun[used_text]) if i != i_other]

            mean_scores = None

            for model in models.values():
                embeddings = model.encode([q] + q_others, normalize_embeddings=True)
                scores = util.dot_score(embeddings[0], embeddings[1:])[0]

                if mean_scores is None:
                    mean_scores = scores
                else:
                    mean_scores += scores

            mean_scores /= len(models)
            idxs = (-mean_scores).argsort()

            count = 0
            last_score = 0.0
            for idx in idxs[:topK]:

                q_idx = q_idxs[idx]
                q_2 = q_others[idx]

                if i not in G:
                    G.add_node(i, text=q)
                if q_idx not in G:
                    G.add_node(q_idx, text=q_2)

                if not G.has_edge(i, q_idx):
                    G.add_edge(i, q_idx, weight=factor * mean_scores[idx])
                else:
                    G[i][q_idx]['weight'] += factor * mean_scores[idx]


def add_text(df_fun, model, G, used_text, topK, idx_to_check, factor, used_embeddings, model_name):
    if topK > 0:
        if used_text not in used_embeddings:
            used_embeddings[used_text] = {}

        if model_name not in used_embeddings[used_text]:
            used_embeddings[used_text][model_name] = []

        for i, q in enumerate(list(df_fun[used_text])):

            q_others = [q_other for i_other, q_other in enumerate(df_fun[used_text]) if i != i_other]
            q_idxs = [i_other for i_other, q_other in enumerate(df_fun[used_text]) if i != i_other]

            query = model.encode(q, normalize_embeddings=True)
            embeddings = model.encode(q_others, normalize_embeddings=True)
            scores = util.dot_score(query, embeddings)[
                0]  # np.array([1 - cosine(query, embeddings[i]) for i in range(len(embeddings))])
            idxs = (-scores).argsort()

            count = 0
            last_score = 0.0
            for idx in idxs:

                q_idx = q_idxs[idx]
                q_2 = q_others[idx]

                if i not in G:
                    G.add_node(i, text=q)
                if q_idx not in G:
                    G.add_node(q_idx, text=q_2)

                if count < topK or scores[idx] == last_score:

                    last_score = scores[idx]

                    # if i == idx_to_check:
                    #    print("-", q_idx, q_2, scores[idx])  # , pairs_sim_scores[idx])

                    if not G.has_edge(i, q_idx):
                        G.add_edge(i, q_idx, weight=factor * scores[idx])
                    else:
                        G[i][q_idx]['weight'] += factor * scores[idx]

                count += 1


def add_text_question(df_fun, model, G, used_text, topK, idx_to_check, factor, used_embeddings, model_name):
    if topK > 0:
        if used_text not in used_embeddings:
            used_embeddings[used_text] = {}

        if model_name not in used_embeddings[used_text]:
            used_embeddings[used_text][model_name] = []

        for i, q in enumerate(list(df_fun[used_text])):

            q_others = [q_other for i_other, q_other in enumerate(df_fun["question"]) if i != i_other]
            q_idxs = [i_other for i_other, q_other in enumerate(df_fun["question"]) if i != i_other]

            query = model.encode(q, normalize_embeddings=True)
            embeddings = model.encode(q_others, normalize_embeddings=True)
            scores = util.dot_score(query, embeddings)[
                0]  # np.array([1 - cosine(query, embeddings[i]) for i in range(len(embeddings))])
            idxs = (-scores).argsort()

            count = 0
            last_score = 0.0
            for idx in idxs:

                q_idx = q_idxs[idx]
                q_2 = q_others[idx]

                if i not in G:
                    G.add_node(i, text=q)
                if q_idx not in G:
                    G.add_node(q_idx, text=q_2)

                if count < topK or scores[idx] == last_score:

                    last_score = scores[idx]

                    if not G.has_edge(i, q_idx):
                        G.add_edge(i, q_idx, weight=factor * scores[idx])
                    else:
                        G[i][q_idx]['weight'] += factor * scores[idx]

                count += 1


def add_text_cross(df_fun, model_sim, G, used_text, topK, idx_to_check, factor, used_embeddings, model_name):
    if topK > 0:
        if used_text not in used_embeddings:
            used_embeddings[used_text] = {}

        if model_name not in used_embeddings[used_text]:
            used_embeddings[used_text][model_name] = []

        for i, q in enumerate(list(df_fun[used_text])):

            pairs_idx = [(i, i_other) for i_other, q_other in enumerate(df_fun[used_text]) if i != i_other]
            pairs = [(q, q_other) for i_other, q_other in enumerate(df_fun[used_text]) if i != i_other]

            candidates_idxs = list(range(len(pairs)))
            pairs_candidates = [pairs[idx] for idx in candidates_idxs]
            pairs_idx_candidates = [pairs_idx[idx] for idx in candidates_idxs]

            scores = model_sim.predict(pairs_candidates)  # + pairs_sim_scores
            idxs = (-scores).argsort()

            count = 0
            last_score = 0.0
            for idx in idxs:

                q_idx = pairs_idx_candidates[idx][1]
                q_2 = pairs_candidates[idx][1]

                if i not in G:
                    G.add_node(i, text=q)
                if q_idx not in G:
                    G.add_node(q_idx, text=q_2)

                if count < topK or scores[idx] == last_score:

                    last_score = scores[idx]
                    if not G.has_edge(i, q_idx):
                        G.add_edge(i, q_idx, weight=factor * scores[idx])
                    else:
                        G[i][q_idx]['weight'] += factor * scores[idx]

                count += 1


def add_text_question_cross(df_fun, model_sim, G, used_text, topK, idx_to_check, factor, used_embeddings, model_name):
    if topK > 0:
        if used_text not in used_embeddings:
            used_embeddings[used_text] = {}

        if model_name not in used_embeddings[used_text]:
            used_embeddings[used_text][model_name] = []

        for i, q in enumerate(list(df_fun[used_text])):

            pairs_idx = [(i, i_other) for i_other, q_other in enumerate(df_fun["question"]) if i != i_other]
            pairs = [(q, q_other) for i_other, q_other in enumerate(df_fun[used_text]) if i != i_other]

            candidates_idxs = list(range(len(pairs)))
            pairs_candidates = [pairs[idx] for idx in candidates_idxs]
            pairs_idx_candidates = [pairs_idx[idx] for idx in candidates_idxs]

            scores = model_sim.predict(pairs_candidates)  # + pairs_sim_scores
            idxs = (-scores).argsort()

            count = 0
            last_score = 0.0
            for idx in idxs:

                q_idx = pairs_idx_candidates[idx][1]
                q_2 = pairs_candidates[idx][1]

                if i not in G:
                    G.add_node(i, text=q)
                if q_idx not in G:
                    G.add_node(q_idx, text=q_2)

                if count < topK or scores[idx] == last_score:

                    last_score = scores[idx]

                    if not G.has_edge(i, q_idx):
                        G.add_edge(i, q_idx, weight=factor * scores[idx])
                    else:
                        G[i][q_idx]['weight'] += factor * scores[idx]

                count += 1


def generate_graph(df_fun, models, model_name, used_embeddings, texts, factors, topK, idx_to_check=92):
    G = nx.DiGraph()

    # 'ref', 'intention', 'reason'
    for text_type, factor in tqdm(zip(texts, factors)):
        if "cross" in model_name:
            model = models[model_name]
            add_text_cross(df_fun, model, G, text_type, topK, idx_to_check, factor, used_embeddings, model_name)
        elif "ensemble" in model_name:
            add_text_ensemble(df_fun, models, G, text_type, topK, idx_to_check, factor, used_embeddings, model_name)
        else:
            model = models[model_name]
            add_text(df_fun, model, G, text_type, topK, idx_to_check, factor, used_embeddings, model_name)

    return G


if __name__ == "__main__":

    df_functions = pd.read_json("../data/functions_enhanced_falcon-7b-instruct.json")
    model_name = "cross-encoder/stsb-roberta-large"
    model_name = "ensemble"
    #model_name = "all-MiniLM-L12-v2"
    #model_name = "quora-distilbert-base"
    #model_name = "all-mpnet-base-v2"

    print(df_functions.head())

    if "cross-encoder" in model_name:
        model_sim = CrossEncoder(model_name, device="cuda:0")
        models = {
            "cross-encoder/stsb-roberta-large": model_sim
        }
    elif "ensemble" in model_name:
        models = {
            "all-MiniLM-L12-v2": SentenceTransformer("all-MiniLM-L12-v2"),
            "quora-distilbert-base": SentenceTransformer('quora-distilbert-base'),
            "all-mpnet-base-v2": SentenceTransformer('all-mpnet-base-v1')
        }
    else:
        models = {
            model_name: SentenceTransformer(model_name),
        }
    #

    used_embeddings = {}

    for topK in [1, 2, 3]:
        filename = f'graphs/{model_name.split("/")[0]}_{topK}_new.pickle'

        if not os.path.isfile(filename):
            G = generate_graph(df_functions, models, model_name, used_embeddings, ["question"], [1.0], topK)
            pickle.dump(G, open(filename, 'wb'))
    print("--")