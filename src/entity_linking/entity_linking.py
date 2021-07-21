import numpy as np

from entity_linking.dataset import EntityLinkingDataset

# args.max_ctxt_len, args.max_title_len, args.max_desc_len
def entity_linking_for_shinradata(
        generator=None,
        ranker=None,
        dataset=None,
        candidate_dataset=None,
        tokenizer=None,
        args=None,
        debug=False,
    ):

    mention_dataset = EntityLinkingDataset(dataset.entity_linking_inputs, tokenizer, args.max_ctxt_len)

    preds, bi_scores, trues, input_ids = generator.generate_candidates(mention_dataset)
    cross_scores, tokens = ranker.predict(
        input_ids, preds, candidate_dataset,
        max_title_len=args.max_title_len,
        max_desc_len=args.max_desc_len)
    rank = np.argsort(np.array(cross_scores), axis=1)[:, ::-1].tolist()
    cross_preds = [[p[s] for s in ss][0] for ss, p in zip(rank, preds)]

    dataset.add_linkpage(cross_preds)

    return dataset
