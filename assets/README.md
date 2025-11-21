## Prompts

### Bootstrap

#### Baseline prompt

These prompts are used with a large language model chat interface to generate content for synthetic genomics reports. This is an intermediate step prior to generation of realistic documents. The output is a table where columns correspond to different document content types, and qualifiers for type of document, content style, and structure. Each row is subsequently given to an LLM to produce fake documents according to the description. This stage therefore introduces variability and coverage of different concepts into the documents being generated.

#### Additional prompt

For each batch (for example - generating 20 rows of data), we present an additional prompt to tailor the output. This could point the batch at a type of test, a disease area, a particular proband pattern, a report style, or any other variable. This can be also used to deliver examples of reports to 'mimic'. This is particularly useful for capturing description of edge cases.

```
"""
Please now generate 20 rows according to the above instructions as a CSV file. These rows should {further instructions}. While conforming to these instructions, please also ensure that rows are varied, and represent a range of different report types and styles.
"""
```

## License

This project uses the CC BY-NC-ND 4.0 license (see [LICENSE](LICENSE)).

The contents of this repository are designed for NHS organisations to use on private data.