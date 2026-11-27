# Figure preparation

The figures in this repository follow a common publication style. The purpose is to make the numerical evidence readable without changing model assumptions, estimation procedures or results.

## Editorial conventions

- Use sentence case. State the statistic or the empirical finding; qualify conclusions by sample, model and uncertainty where relevant.
- Identify units, sample period and data source. Distinguish observed, fitted, projected and simulated values.
- Reserve red for losses, breaches or adverse scenarios. Use line styles or markers as well as colour where identification matters. The categorical palette is not a substitute for a colour-vision accessibility check.
- Use a zero baseline for magnitude comparisons in bar charts. Identify non-linear axes explicitly. Use a common scale for comparable panels.
- State the confidence level and definition of uncertainty bands. Identify in-sample and out-of-sample observations.
- Avoid promotional language, decorative icons, gradients and unsupported claims. Use British English consistently, except in established technical terms.

## Export

The existing `readme_figures` command continues to produce the two PNG files expected by the README and existing tests. The default width is 2,400 pixels; the following environment variables are optional:

```powershell
$env:RESEARCH_CHART_WIDTH_PX = '3840'
$env:RESEARCH_CHART_SVG = '1'
```

For validation, run the generator with `--synthetic --out` pointing to a temporary directory. Never replace published official-data figures with synthetic figures. After checking the previews, regenerate using official data. Inspect legends, long labels and notes at GitHub display size; the automated fit check covers figure-level text only, not all overlapping axes artists.

## References

- Office for National Statistics, [Data visualisation guidance](https://service-manual.ons.gov.uk/data-visualisation/guidance).
- Office for National Statistics, [Using colours in charts](https://service-manual.ons.gov.uk/data-visualisation/colours/using-colours-in-charts).
- European Union, [Data visualisation guide](https://data.europa.eu/apps/data-visualisation-guide/).
- Financial Times Visual Journalism Team, [Visual Vocabulary](https://github.com/Financial-Times/chart-doctor/tree/main/visual-vocabulary).

These references inform editorial conventions; the palette and implementation are independent.
