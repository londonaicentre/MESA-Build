## Bootstrap Table Instructions
The prompts below are used with a large language model chat interface to generate content for synthetic genomics reports. This is an intermediate step prior to generation of realistic documents. The output is a table where columns correspond to different document content types, and qualifiers for type of document, content style, and structure. Each row is subsequently given to an LLM to produce fake documents according to the description. This stage therefore introduces variability and coverage of different concepts into the documents being generated.

### Baseline prompt
```
"""
You are an expert genomic medicine doctor and clinical informatician with extensive experience in medical documentation, genetic testing, and data structuring. Take note of the included pydantic schema for structuring genomic report information for the type of information that should be included.

**Task Overview**
Create a comprehensive table that summarises genomic test reports in broad categories. This table should contain enough information that each row can be subsequently used to generate synthetic reports and structured schemas for a genomic testing information system.

**Table Requirements**
Generate a table with the following columns, ensuring diverse and representative examples across all test types:
Test Type - The specific genomic test type (DNA, FISH, Karyotype, PCR, MLPA, Other)
Test Details - Methodology, coverage, specific analysis techniques
Result Entit(ies) - All entit(ies) (e.g. Gene, Variant, Chromosome, etc.) including specific names
Result Description - Detailed technical description of findings as would appear in a report, including abnormality status, and classification of quantitative findings if necessary. There may be multiple findings for different entities.
Clinical Context - Patient presentation, referral reason, and rationale for testing
Disease Context - Suspected or confirmed conditions related to the test
Family History - Relevant genetic information about relatives, if any
Proband Info - Whose sample was tested (patient, relative, etc.)
Clinical Implications - Interpretation of results for patient care
Recommendations - Follow-up testing, referrals, or management suggestions
Report style - This table will be used to generate synthetic reports. Describe a report style, that may include elements such as is conciseness, structure, and verbosity of clinical descriptions.

The table should be structured as a csv file with the following headings:
test_type,test_details,result_entities,result_description,clinical_context,disease_context,family_history,test_subject,clinical_implications,recommendations,report_style

**Further Instructions**
- Each example may contain multiple entities or results.
- Please make results clinically consistent.
- Most rows will have missing content in some fields. For example, concise reports may exclude most context and history, and just report the result.

Please confirm that you understand these instructions.
"""
```


### Additional prompt
For each batch (for example - generating 20 rows of data), we present an additional prompt to tailor the output. This could point the batch at a type of test, a disease area, a particular proband pattern, a report style, or any other variable. This can be also used to deliver examples of reports to 'mimic'. This is particularly useful for capturing description of edge cases.

```
"""
Please now generate 20 rows according to the above instructions as a CSV file. These rows should {further instructions}. While conforming to these instructions, please also ensure that rows are varied, and represent a range of different report types and styles.
"""
```
