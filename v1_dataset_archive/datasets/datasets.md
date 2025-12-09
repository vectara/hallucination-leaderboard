# Datasets used in HHEM Leaderboard

Please note that in HHEM leaderboard, we only use the text source and nothing else (neither LLM's output nor human annotation) from each of the following datasets.

Version 1, released in October 2023: 

- [MNBM](https://github.com/google-research-datasets/xsum_hallucination_annotations)
  - num. of samples: 273
  - Text source dataset: XSum
  - Notes: 
    - Google's [True](https://aclanthology.org/2022.naacl-main.287.pdf) benchmark calls it "MNBM" while many other works call it "XSumFaith"
    - 500 articles from XSum were covered in the original paper. So the LB used about half of it. 

- [FEVER dev split](https://aclanthology.org/W18-5501/)
  - num. of samples: 260
  - Text source dataset: Wikipedia
  - Notes: The context is from Wikipedia while the claim is added by FEVER. 

- `Summac test_polytope`, the test split of the dataset PolyTope from the [Summac](https://arxiv.org/pdf/2111.09525) benchmark
  - num. of samples: 150
  - Text source dataset: CNN/DM
  - Notes: Ploytope randomly picked 150 samples from CNN/DM (unclear which split) for human annovation. So LB used all of them here. 

- `vitc_dev`, the dev split of [VitaminC](https://aclanthology.org/2021.naacl-main.52.pdf)
  - num. of samples: 113
  - Text source dataset: Wikipedia. There are three parts of VitaminC:
    - 50k most viewed English Wikipedia articles as of January 2020. 
    - All articles from FEVER. 
    - All COVID-19 articles categorized by Wikimedia Foundation. (It is unclear how big is this part)
    - Also synthesized claim-evidance pairs (unclear whether they are included in HHEM LB).
  
- `summeval_valid`, the dev split of [SummEval](https://arxiv.org/pdf/2007.12626)
  - num. of samples: 100
  - Text source dataset: CNN/DM's test set
  - Notes: They randomly picked 100 samples from CNN/DM's test set for human annotation. So LB used all of them. 

- `frank_valid`, the validation split of [Frank](https://arxiv.org/pdf/2109.08602)
  - num. of samples: 59
  - Text source dataset: Both CNN/DM and XSum
  - Notes: [The validation split of Frank has 149 articles](https://github.com/artidoro/frank?tab=readme-ov-file#validation-test-split-for-frank). So LB used about half of it. 

XSUM and CNN/DM:
* [XSum](https://aclanthology.org/D18-1206.pdf) is built on 226,711 BBC articles from 2010 to 2017, covering topics News, Politics, Sports, Weather, Business, Technology, Science, Health, Family, Education, Entertainment and Arts. It's training/validation/test split is 90/5/5.
* CNN/DM, perhaps the most famoust dataset in NLP, it has 93k articles from CNN and 220k from Daily Mail. 