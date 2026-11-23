# Synthetic Clinical Note Generation

## NHS England Data Science and Applied AI Team

![Robot doctor writing synthetic clinical notes](docs/pics/SCN_banner.jpg)

This project uses LLMs to generate synthetic clinical notes for entire patient journeys in hospitals.

### ⚠️ Important Notice to Users ⚠️

All data found in this repository is entirely **synthetic**.

Synthetic data is artificially generated data that mimics real-world data. It is typically created using real data as a seed and adding noise. However, in this pipeline **no real data is used** at any point. Synthetic data can help with analysis, testing, or model training without using real data.

Synthetic data does have limitations. For more information please read `docs/synthetic_data_limitations`.

### What does this project do?

This pipeline was developed to aid the testing and evaluation of AI generated discharge summaries.

Using OpenAI's `gpt-4o`, this pipeline generated **high quality** and **realistic** patient journeys and clinical notes.

Clinicians were heavily involved in the evaluation of clinical notes from this pipeline. Their thorough feedback was used to iteratively improve the pipeline.

**The pipeline:**

- Generates synthetic patients.
- Generates realistic admission reasons (emergency or elective) for each patient.
- Generates a realistic patient journey from the point of admission to just before discharge.
- Generates realistic clinical notes for each stage of the journey.
- Adds augmentations to each note (typos and medical abbreviations)

The pipeline is highly configurable using `config/params.py` and `config/config.py`.

Whilst the project was developed on Foundry, it was designed to be easily adaptable to other platforms.

The local pipeline reads and writes CSV files and can call OpenAI or run a model locally with vLLM.

The pipeline was tested with Python 3.12.12.

### Getting Started

1. Read the `docs` to better understand what input data needs to be inputted into the pipeline.
2. Configure the model in `config/params.py`. For local inference, follow the vLLM setup below.
3. Install necessary libraries. These can be found in `requirements.txt` but may need to be installed differently depending on your platform.
4. Check `config/config.py` and `config/params.py`. For more info see: `docs/adapting_the_pipeline`.
5. Check `src/dataset_utils.py`, `src/doc_templates.py`, `src/prompts.py` and `src/schemas.py`. For more info see: `docs/adapting_the_pipeline`.
6. Run `notebooks/run_pipeline.ipynb`.

### Dependencies

This project did originally make LLM calls via an API to a deployed version of OpenAI's `gpt-4o` model.
See [https://openai.com/policies/service-terms](https://openai.com/policies/service-terms) and [https://platform.openai.com/](https://platform.openai.com/) for further details.

### Run a local model with vLLM

Install the `vllm` Python package in the notebook's environment on a supported Linux host with enough memory for the model. Follow the [vLLM installation instructions](https://docs.vllm.ai/en/stable/getting_started/installation/) for your hardware.

Set `model` in `config/params.py` to `vllm:Qwen/Qwen3.5-27B` before running the notebook. The `vllm:` prefix also accepts a local model path. The pipeline loads the model in Python and uses its chat template for each conversation. For Qwen3.5, the pipeline disables thinking mode so the response contains the requested output directly. [vLLM offline chat example](https://docs.vllm.ai/en/stable/examples/basic/offline_inference/)

Set `VLLM_TENSOR_PARALLEL_SIZE` to split the model across multiple GPUs, or `VLLM_MAX_MODEL_LEN` to set a context limit that fits your hardware. For example, set `VLLM_MAX_MODEL_LEN=32768` to cap the context at 32,768 tokens. `VLLM_MAX_TOKENS` caps generated tokens per response and defaults to `8192`.

### Patient JSON output

The final notebook step writes one single-line JSON record per patient under `data/patient_json/<version_tag>/`. Its fields are `patient_id`, `success_flag`, `history`, `label`, `target_text`, `target_visit_time`, and `accepted_visit_count`. The `history` string shows admission and discharge times followed by the event notes in journey order. If a discharge timestamp is unavailable, the export uses the final journey event time.

The current generator samples one admission per patient. If any event note is missing or failed, `history` is empty, `accepted_visit_count` is `0`, and `success_flag` is `PS`. The NSDUH target assessment is scheduled after the 12-month history window, at month 13 from the trajectory index date. Legacy inputs without a trajectory index date or visit months use `null` for `target_visit_time`. The final step also writes the evaluation CSVs.

### Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

_See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidance._

## License

Unless stated otherwise, the codebase is released under [the MIT License][mit].
This covers both the codebase and any sample code in the documentation.

_See [LICENSE](./LICENSE) for more information._

The documentation is [© Crown copyright][copyright] and available under the terms
of the [Open Government 3.0][ogl] licence.

[mit]: LICENCE
[copyright]: http://www.nationalarchives.gov.uk/information-management/re-using-public-sector-information/uk-government-licensing-framework/crown-copyright/
[ogl]: http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/

### Contributors (Alphabetical)

- Alice Waterhouse
- Amaia Imaz Blanco
- Ben Wallace
- Jonny Pearson
- Michael Spence
- Mobolu Olowoyeye
- Scarlett Kynoch
- Will Poulett

If you have questions, please [contact us](mailto:england.datascience@nhs.net).

### Cite this work:

If using this work, please cite our data paper:

```
@misc{poulett2026pipelinegeneratinglongitudinalsynthetic,
      title={A Pipeline for Generating Longitudinal Synthetic Clinical Notes Using Large Language Models}, 
      author={William Poulett and Alice Waterhouse and Ben Wallace and Scarlett Kynoch and Amaia Imaz Blanco and Michael Spence and Jonathan Pearson},
      year={2026},
      eprint={2606.26879},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2606.26879}, 
}
```
