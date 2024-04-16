# KoDialogBench

### Paper

Title: KoDialogBench: Evaluating Conversational Understanding of Language Models with Korean Dialogue Benchmark

Abstract: [Paper link](https://arxiv.org/abs/2402.17377)

As language models are often deployed as chatbot assistants, it becomes a virtue for models to engage in conversations in a user's first language. While these models are trained on a wide range of languages, a comprehensive evaluation of their proficiency in low-resource languages such as Korean has been lacking. In this work, we introduce KoDialogBench, a benchmark designed to assess language models' conversational capabilities in Korean. To this end, we collect native Korean dialogues on daily topics from public sources, or translate dialogues from other languages. We then structure these conversations into diverse test datasets, spanning from dialogue comprehension to response selection tasks. Leveraging the proposed benchmark, we conduct extensive evaluations and analyses of various language models to measure a foundational understanding of Korean dialogues. Experimental results indicate that there exists significant room for improvement in models' conversation skills. Furthermore, our in-depth comparisons across different language models highlight the effectiveness of recent training techniques in enhancing conversational proficiency. We anticipate that KoDialogBench will promote the progress towards conversation-aware Korean language models.

Homepage: [GitHub repository](https://github.com/sb-jang/kodialogbench)


### Citation

```bibtex
@misc{jang2024kodialogbench,
      title={KoDialogBench: Evaluating Conversational Understanding of Language Models with Korean Dialogue Benchmark}, 
      author={Seongbo Jang and Seonghyeon Lee and Hwanjo Yu},
      year={2024},
      eprint={2402.17377},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Groups and Tasks

#### Groups

* `kodialogbench`: All 21 tasks of KoDialogBench including dialogue comprehension and response selection
* `kodialogbench_dc`: 14 tasks of dialogue comprehension categorized by 6 aspects (topic, emotion, relation, location, dialog act, fact)
* `kodialogbench_rs`: 7 tasks of response selection from different sources
* `kodialogbench_dc_topic`: 3 topic tasks
* `kodialogbench_dc_emotion`: 3 emotion tasks
* `kodialogbench_dc_relation`: 2 relation tasks
* `kodialogbench_dc_dialog_act`: 2 dialog act tasks
* `kodialogbench_dc_fact`: 3 fact tasks

#### Tasks

All tasks consist of multiple-choice questions.
* `kodialogbench_dc_topic_k-sns`: 6 options, 1200 examples
* `kodialogbench_dc_topic_k-tdd`: 19 options, 1900 examples
* `kodialogbench_dc_topic_socialdial`: 4 options, 400 examples
* `kodialogbench_dc_emotion_k-ed`: 6 options, 1200 examples
* `kodialogbench_dc_emotion_dailydialog`: 5 options, 470 examples
* `kodialogbench_dc_emotion_empathetic`: 2 options, 2000 examples
* `kodialogbench_dc_relation_socialdial-distance`: 4 options, 524 examples
* `kodialogbench_dc_relation_socialdial-relation`: 3 options, 330 examples
* `kodialogbench_dc_location_socialdial`: 4 options, 376 examples
* `kodialogbench_dc_dialog_act_k-tdd`: 4 options, 520 examples
* `kodialogbench_dc_dialog_act_dailydialog`: 4 options, 1000 examples
* `kodialogbench_dc_fact_k-ds`: 4 options, 1200 examples
* `kodialogbench_dc_fact_personachat`: 4 options, 1000 examples
* `kodialogbench_dc_fact_empathetic`: 4 options, 2394 examples

Response selection tasks provide 5 options.
* `kodialogbench_rs_k-sns`: 10295 examples
* `kodialogbench_rs_k-tdd`: 10616 examples
* `kodialogbench_rs_k-ed`: 17818 examples
* `kodialogbench_rs_personachat`: 7801 examples
* `kodialogbench_rs_dailydialog`: 6740 examples
* `kodialogbench_rs_empathetic`: 7941 examples
* `kodialogbench_rs_socialdial`: 7237 examples

### Checklist

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?
