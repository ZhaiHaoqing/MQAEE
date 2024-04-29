# AI6103 Project: Reimplementation of MQAEE

This repo is created for the AI6103 Deep Learning & Applications project. It includes an unofficial implementation of the paper [Event Extraction as Multi-turn Question Answering](https://aclanthology.org/2020.findings-emnlp.73/) by Li et al.

The code is based on [TextEE](https://github.com/ej0cl6/TextEE/).

**Author**: Zhai Haoqing

**Group members**: Pham Minh Khoi, Siew Tze Kang, Su Su Yi, Sui Jinhong

<table  style="width:100%" border="0">
<thead>
<tr class="header">
  <th><strong> Name </strong></th>
  <th><strong> Matriculation number</strong></th>
  <th><strong> Email </strong></th>
</tr>
</thead>
<tbody>
<tr>
  <td> Pham Minh Khoi </td>
  <td> G2303923E </td>
  <td> s230119@e.ntu.edu.sg </td>
</tr>
<tr>
  <td> Siew Tze Kang, Julian </td>
  <td> G2303643F </td>
  <td> siew0056@e.ntu.edu.sg </td>
</tr>
<tr>
  <td> Su Su Yi </td>
  <td> G2304221L </td>
  <td> susu001@e.ntu.edu.sg </td>
</tr>
<tr>
  <td> Sui Jinhong </td>
  <td> G2302859J </td>
  <td> suij0003@e.ntu.edu.sg </td>
</tr>
<tr>
  <td> Zhai Haoqing </td>
  <td> G2303503A </td>
  <td> zhai0032@e.ntu.edu.sg </td>
</tr>
</tbody>
</table>


## Introduction
To fulfill the requirements of the project, we have chosen a 2020 conference paper to reimplement its algorithm. This algorithm,
referred to as MQAEE, splits event extraction into three sub-tasks: trigger identification, trigger classification, and argument extraction. These subtasks are further modeled by a series of machine reading comprehension (MRC) based QA templates. Please check more details in the original paper [Event Extraction as Multi-turn Question Answering](https://aclanthology.org/2020.findings-emnlp.73/).

## Reimplementation Results
<table  style="width:100%" border="0">
<thead>
<tr class="header">
  <th><strong> </strong></th>
  <th><strong>Tri-Id F1(%)</strong></th>
  <th><strong>Tri-Cls F1(%)</strong></th>
  <th><strong>Arg-Id F1(%)</strong></th>
  <th><strong>Arg-Cls F1(%)</strong></th>
</tr>
</thead>
<tbody>
<tr>
  <td> Official </td>
  <td> 74.5 </td>
  <td> 71.7 </td>
  <td> 55.2 </td>
  <td> 53.4 </td>
</tr>
<tr>
  <td> Reproduce </td>
  <td> 70.9 </td>
  <td> 54.0 </td>
  <td> 47.8 </td>
  <td> 45.6 </td>
</tr>
</tbody>
</table>

## Environment
Please install the following packages from both conda and pip.

```
conda install
  - python 3.8
  - pytorch 2.0.1
  - numpy 1.24.3
  - ipdb 0.13.13
  - tqdm 4.65.0
  - beautifulsoup4 4.11.1
  - lxml 4.9.1
  - jsonlines 3.1.0
  - jsonnet 0.20.0
  - stanza=1.5.0
```
```
pip install
  - transformers 4.30.0
  - sentencepiece 0.1.96
  - scipy 1.5.4
  - spacy 3.1.4
  - nltk 3.8.1
  - tensorboardX 2.6
  - keras-preprocessing 1.1.2
  - keras 2.4.3
  - dgl-cu111 0.6.1
  - amrlib 0.7.1
  - cached_property 1.5.2
  - typing-extensions 4.4.0
  - penman==1.2.2
```

## Training
```
python train.py -c [config]
```

## Evaluation
```
python evaluate_pipeline.py --data [data_path] --triid_model [model_path] --tricls_model [model_path] --argext_model [model_path]
```
You can download pre-trained models from [Google Drive](https://drive.google.com/drive/folders/1dlDoz0-8KaYNOE1DUZNa7OlSkkcZAQ41?usp=drive_link) to the outputs folder.
## Contact
If you have any question, please email zhai0032@e.ntu.edu.sg.