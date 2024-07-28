---
base_model: onlplab/alephbert-base
datasets: []
language: []
library_name: sentence-transformers
metrics:
- pearson_cosine
- spearman_cosine
- pearson_manhattan
- spearman_manhattan
- pearson_euclidean
- spearman_euclidean
- pearson_dot
- spearman_dot
- pearson_max
- spearman_max
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:300067
- loss:CosineSimilarityLoss
widget:
- source_sentence: רוב מחקרי המקרה הם
  sentences:
  - המנהיגות חייבת להכיר בכך שאסטרטגיית כוח עבודה גמישה היא הכרחית.
  - לוריין בראקו הייתה רהוטה ודינמית.
  - לרוב המקרים המחקריים יש משהו במשותף.
- source_sentence: טוב קלס דיבר ראשון.
  sentences:
  - 'הפחתת הגירעון היא מאוד פשוטה. '
  - המודעות זהירות ומתחשבות ככל האפשר.
  - טוב קל היה הראשון שדיבר.
- source_sentence: אני אסיר תודה לאלו שקראו אותו.
  sentences:
  - 'אני לא אוהב אף אחד שקרא ואני לא מודה. '
  - וואקאיאמה ידועה לשמצה כבעלת מרכזי הקניות הטובים ביותר.
  - האישה נרצחה ממש מחוץ לחדר השינה שלה.
- source_sentence: ההחלטה ברוב של 6-3 קבעה שכאלה הגבלות לא מפרות את זכויות התיקון
    הראשון, שכן כסף הוא רכוש; הוא אינו דיבור.
  sentences:
  - אוויטה הייתה מאוד רזה.
  - מועצת העיר הייתה מקום שבו כל אזרח יכול היה לקחת חלק במינהל העיר.
  - חופש הדיבור כבר לא קיים באמריקה.
- source_sentence: עברתי הרבה בחיים שלי, אבל הדברים שסוזן ראתה הפחידו אותי בטירוף.
  sentences:
  - ללקוחות זמין מגוון רחב יותר של שירותים מאשר היה פעם.
  - מחוקקים יסתכלו על נושאים חדשים בחשדנות.
  - 'סוזן ראתה דברים שיותר מפחידים אותי מכל דבר שראיתי אי פעם. '
model-index:
- name: SentenceTransformer based on onlplab/alephbert-base
  results:
  - task:
      type: semantic-similarity
      name: Semantic Similarity
    dataset:
      name: sts dev
      type: sts-dev
    metrics:
    - type: pearson_cosine
      value: 0.1533334668637415
      name: Pearson Cosine
    - type: spearman_cosine
      value: 0.13198303334972783
      name: Spearman Cosine
    - type: pearson_manhattan
      value: 0.1599405179006303
      name: Pearson Manhattan
    - type: spearman_manhattan
      value: 0.13096803910222896
      name: Spearman Manhattan
    - type: pearson_euclidean
      value: 0.1572174385303871
      name: Pearson Euclidean
    - type: spearman_euclidean
      value: 0.13154539267601986
      name: Spearman Euclidean
    - type: pearson_dot
      value: 0.17004907521956014
      name: Pearson Dot
    - type: spearman_dot
      value: 0.15716660762775467
      name: Spearman Dot
    - type: pearson_max
      value: 0.17004907521956014
      name: Pearson Max
    - type: spearman_max
      value: 0.15716660762775467
      name: Spearman Max
    - type: pearson_cosine
      value: 0.23175076559950836
      name: Pearson Cosine
    - type: spearman_cosine
      value: 0.10472125124251425
      name: Spearman Cosine
    - type: pearson_manhattan
      value: 0.22047573056124203
      name: Pearson Manhattan
    - type: spearman_manhattan
      value: 0.11522327796756969
      name: Spearman Manhattan
    - type: pearson_euclidean
      value: 0.2140070132697967
      name: Pearson Euclidean
    - type: spearman_euclidean
      value: 0.10462358622897888
      name: Spearman Euclidean
    - type: pearson_dot
      value: 0.23556059744792593
      name: Pearson Dot
    - type: spearman_dot
      value: 0.14065174113727108
      name: Spearman Dot
    - type: pearson_max
      value: 0.23556059744792593
      name: Pearson Max
    - type: spearman_max
      value: 0.14065174113727108
      name: Spearman Max
---

# SentenceTransformer based on onlplab/alephbert-base

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [onlplab/alephbert-base](https://huggingface.co/onlplab/alephbert-base). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [onlplab/alephbert-base](https://huggingface.co/onlplab/alephbert-base) <!-- at revision 1745fb3ff5137e41e9eb4d6246e0758f63b93e46 -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 tokens
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'עברתי הרבה בחיים שלי, אבל הדברים שסוזן ראתה הפחידו אותי בטירוף.',
    'סוזן ראתה דברים שיותר מפחידים אותי מכל דבר שראיתי אי פעם. ',
    'מחוקקים יסתכלו על נושאים חדשים בחשדנות.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Semantic Similarity
* Dataset: `sts-dev`
* Evaluated with [<code>EmbeddingSimilarityEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator)

| Metric             | Value      |
|:-------------------|:-----------|
| pearson_cosine     | 0.1533     |
| spearman_cosine    | 0.132      |
| pearson_manhattan  | 0.1599     |
| spearman_manhattan | 0.131      |
| pearson_euclidean  | 0.1572     |
| spearman_euclidean | 0.1315     |
| pearson_dot        | 0.17       |
| spearman_dot       | 0.1572     |
| pearson_max        | 0.17       |
| **spearman_max**   | **0.1572** |

#### Semantic Similarity
* Dataset: `sts-dev`
* Evaluated with [<code>EmbeddingSimilarityEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator)

| Metric             | Value      |
|:-------------------|:-----------|
| pearson_cosine     | 0.2318     |
| spearman_cosine    | 0.1047     |
| pearson_manhattan  | 0.2205     |
| spearman_manhattan | 0.1152     |
| pearson_euclidean  | 0.214      |
| spearman_euclidean | 0.1046     |
| pearson_dot        | 0.2356     |
| spearman_dot       | 0.1407     |
| pearson_max        | 0.2356     |
| **spearman_max**   | **0.1407** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset


* Size: 300,067 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                        | label                                                              |
  |:--------|:-----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:-------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                            | int                                                                |
  | details | <ul><li>min: 5 tokens</li><li>mean: 23.13 tokens</li><li>max: 112 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 12.52 tokens</li><li>max: 35 tokens</li></ul> | <ul><li>0: ~34.40%</li><li>1: ~34.70%</li><li>2: ~30.90%</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                            | sentence_1                                                         | label          |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------|:---------------|
  | <code>טריפ טוענת שהיא עשתה את הקלטות כדי להגן על עצמה כי לוינסקי לחצה עליה לשקר בתיק פאולה ג'ונס.</code>                                                                              | <code>הקלטות שימשו להגן על טריפ מפני לוינסקי.</code>               | <code>0</code> |
  | <code>מספר הנחיות מעצבות את מאמצינו.</code>                                                                                                                                           | <code>הנחיות אלו נכתבו על ידי מייסדנו לאחר שחלם חלום מוזר. </code> | <code>1</code> |
  | <code>רק בימים האחרונים ב"ניו יורק טיימס", וולטר גודמן אפיין את קרוספייר כתוכנית הצעקות של CNN, ומורין דאוד סיכמה את חובותיה של פרארו כפטפוט לילה אחר לילה עם האקרים פוליטיים.</code> | <code>וולטר גודמן מעולם לא ראה את קרוספייר.</code>                 | <code>2</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 1
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step  | Training Loss | sts-dev_spearman_max |
|:------:|:-----:|:-------------:|:--------------------:|
| 0.0267 | 500   | 0.7406        | -                    |
| 0.0533 | 1000  | 0.673         | -0.0488              |
| 0.0800 | 1500  | 0.677         | -                    |
| 0.1066 | 2000  | 0.6694        | -0.0525              |
| 0.1333 | 2500  | 0.6682        | -                    |
| 0.1600 | 3000  | 0.6527        | -0.0462              |
| 0.1866 | 3500  | 0.6709        | -                    |
| 0.2133 | 4000  | 0.6669        | -0.0463              |
| 0.2399 | 4500  | 0.6678        | -                    |
| 0.2666 | 5000  | 0.663         | -0.0464              |
| 0.2933 | 5500  | 0.6678        | -                    |
| 0.3199 | 6000  | 0.6725        | -0.0464              |
| 0.3466 | 6500  | 0.664         | -                    |
| 0.3732 | 7000  | 0.6816        | -0.0470              |
| 0.3999 | 7500  | 0.6684        | -                    |
| 0.4266 | 8000  | 0.6766        | -0.0488              |
| 0.4532 | 8500  | 0.67          | -                    |
| 0.4799 | 9000  | 0.6652        | -0.0483              |
| 0.5065 | 9500  | 0.6745        | -                    |
| 0.5332 | 10000 | 0.667         | -0.0502              |
| 0.5599 | 10500 | 0.6662        | -                    |
| 0.5865 | 11000 | 0.6709        | -0.0483              |
| 0.6132 | 11500 | 0.6789        | -                    |
| 0.6398 | 12000 | 0.6672        | -0.0477              |
| 0.6665 | 12500 | 0.6718        | -                    |
| 0.6931 | 13000 | 0.6571        | -0.0494              |
| 0.7198 | 13500 | 0.6722        | -                    |
| 0.7465 | 14000 | 0.6677        | -0.0487              |
| 0.7731 | 14500 | 0.6584        | -                    |
| 0.7998 | 15000 | 0.6698        | -0.0490              |
| 0.8264 | 15500 | 0.6602        | -                    |
| 0.8531 | 16000 | 0.664         | 0.1407               |
| 0.8798 | 16500 | 0.668         | -                    |
| 0.9064 | 17000 | 0.6595        | 0.1457               |
| 0.9331 | 17500 | 0.6529        | -                    |
| 0.9597 | 18000 | 0.6612        | 0.1483               |
| 0.9864 | 18500 | 0.6589        | -                    |
| 1.0    | 18755 | -             | 0.1572               |
| 0.0267 | 500   | 0.6548        | -                    |
| 0.0533 | 1000  | 0.6571        | 0.1268               |
| 0.0800 | 1500  | 0.6634        | -                    |
| 0.1066 | 2000  | 0.6512        | 0.1487               |
| 0.1333 | 2500  | 0.6509        | -                    |
| 0.1600 | 3000  | 0.6527        | 0.1369               |
| 0.1866 | 3500  | 0.6573        | -                    |
| 0.2133 | 4000  | 0.6433        | 0.1297               |
| 0.2399 | 4500  | 0.6576        | -                    |
| 0.2666 | 5000  | 0.6555        | 0.1394               |
| 0.2933 | 5500  | 0.6391        | -                    |
| 0.3199 | 6000  | 0.6495        | 0.1543               |
| 0.3466 | 6500  | 0.6626        | -                    |
| 0.3732 | 7000  | 0.6474        | 0.1198               |
| 0.3999 | 7500  | 0.6577        | -                    |
| 0.4266 | 8000  | 0.64          | 0.1449               |
| 0.4532 | 8500  | 0.6508        | -                    |
| 0.4799 | 9000  | 0.6533        | 0.1670               |
| 0.5065 | 9500  | 0.6419        | -                    |
| 0.5332 | 10000 | 0.6505        | 0.1466               |
| 0.5599 | 10500 | 0.6475        | -                    |
| 0.5865 | 11000 | 0.643         | 0.1269               |
| 0.6132 | 11500 | 0.6427        | -                    |
| 0.6398 | 12000 | 0.6444        | 0.1516               |
| 0.6665 | 12500 | 0.6523        | -                    |
| 0.6931 | 13000 | 0.6408        | 0.1400               |
| 0.7198 | 13500 | 0.6343        | -                    |
| 0.7465 | 14000 | 0.6347        | 0.1373               |
| 0.7731 | 14500 | 0.6364        | -                    |
| 0.7998 | 15000 | 0.6372        | 0.1424               |
| 0.8264 | 15500 | 0.6496        | -                    |
| 0.8531 | 16000 | 0.641         | 0.1690               |
| 0.8798 | 16500 | 0.6504        | -                    |
| 0.9064 | 17000 | 0.641         | 0.1350               |
| 0.9331 | 17500 | 0.6495        | -                    |
| 0.9597 | 18000 | 0.6338        | 0.1405               |
| 0.9864 | 18500 | 0.6472        | -                    |
| 1.0    | 18755 | -             | 0.1407               |


### Framework Versions
- Python: 3.10.12
- Sentence Transformers: 3.0.1
- Transformers: 4.42.4
- PyTorch: 2.3.1+cu121
- Accelerate: 0.32.1
- Datasets: 2.20.0
- Tokenizers: 0.19.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->