# Trace Sorter Methods Comparison

- Trace path: `D:\Data\agent\trace\all`
- Metadata: `D:\Code\github\hehe03\skills-repo\高交all.csv`
- Train source: `trace_path split=train`
- Train samples: 21
- Test source: `trace_path split=test`
- Test samples: 88
- Methods: non_llm_no_train, non_llm_unlabeled, non_llm_labeled
- Aux components: `all`
- Ensemble policy: `precision_guard`

## Method Structure

| method | family | training scenario | rules loaded | generated rule files |
|---|---|---|---:|---|
| non_llm_no_train | non_llm | no_train | 13 |  |
| non_llm_unlabeled | non_llm | unlabeled | 88 | `D:\Code\github\hehe03\skills-repo\TraceSorter\scripts\rules\dynamic\non_llm\unlabeled_rules.json` |
| non_llm_labeled | non_llm | labeled | 69 | `D:\Code\github\hehe03\skills-repo\TraceSorter\scripts\rules\dynamic\non_llm\labeled_rules.json` |

## Method Notes

Final answer policy:
| policy | samples |
|---|---:|
| `none/none:none` | 88 |

## Metrics

| method | rules | accuracy | precision(badcase) | recall(badcase) | f1(badcase) | tp | fp | tn | fn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| non_llm_no_train | 13 | 0.7386 | 0.0 | 0.0 | 0.0 | 0 | 0 | 65 | 23 |
| non_llm_unlabeled | 88 | 0.2614 | 0.2614 | 1.0 | 0.4144 | 23 | 65 | 0 | 0 |
| non_llm_labeled | 69 | 0.7159 | 0.4667 | 0.6087 | 0.5283 | 14 | 16 | 49 | 9 |

## Predictions

| name | label | non_llm_no_train | non_llm_unlabeled | non_llm_labeled |
|---|---|---|---|---|
| `0250408李俊风总拜访上海交通大学6-副本.json` | badcase | goodcase (bad=0.0, good=0.5) | badcase (bad=4.24, good=0.5) | goodcase (bad=4.36, good=4.41) |
| `0426f4ae3329f7e18b8b7c5e5955aa22.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=0.68, good=0.5) | goodcase (bad=0.83, good=3.971) |
| `06ca7aedb386728876b8181c2807281c.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=0.68, good=0.5) | goodcase (bad=0.35, good=4.4725) |
| `099e97a039bc343d03f9a19878bb2c61.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=1.31, good=0.5) | goodcase (bad=0.83, good=3.9563) |
| `0be798db04da11b35aea0b0b7879fa1b.json` | goodcase | goodcase (bad=0.45, good=0.25) | badcase (bad=1.38, good=0.25) | goodcase (bad=1.04, good=3.6782) |
| `0de9ffb66ff1e6509d0a30422bcc275a.json` | badcase | goodcase (bad=0.45, good=0.25) | badcase (bad=3.2, good=0.25) | badcase (bad=5.29, good=3.8) |
| `13f5cf0c2b3da8d4c577469eeaaf2c73.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=2.86, good=0.5) | goodcase (bad=3.89, good=4.29) |
| `169c94c4290053d811be8da0c4d2aae6.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=0.68, good=0.5) | goodcase (bad=0.83, good=3.971) |
| `1f0fc98ceb2a2d9bea3ae53e10243274.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=0.68, good=0.5) | goodcase (bad=0.35, good=4.4725) |
| `1fd2438a5afee415642a33ef85cc6dce.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=1.11, good=0.5) | goodcase (bad=0.35, good=4.4802) |
| `20241204梁博拜访上海交通大学V2-7-副本.json` | badcase | goodcase (bad=0.0, good=0.5) | badcase (bad=4.52, good=0.5) | badcase (bad=6.6706, good=0.5) |
| `20241204梁博拜访上海交通大学V2-7.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=4.24, good=0.5) | goodcase (bad=4.36, good=4.41) |
| `20250407张熙伟总拜访上海交通大学V2-7.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=3.11, good=0.5) | goodcase (bad=4.24, good=4.49) |
| `25676385006ee1e8c06c02477c369a61.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=1.31, good=0.5) | goodcase (bad=0.59, good=3.9483) |
| `27aae3045288bcc9585e0588b8c38a01.json` | badcase | goodcase (bad=0.0, good=0.5) | badcase (bad=4.02, good=0.5) | badcase (bad=6.3152, good=0.5) |
| `2c60dacd9518141fb2da9402740f0f63-策划报告.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=1.3, good=0.5) | goodcase (bad=0.49, good=3.68) |
| `2eea06de8b19a7b9f063cb9b729b65fb.json` | goodcase | goodcase (bad=0.45, good=0.25) | badcase (bad=3.06, good=0.25) | badcase (bad=4.93, good=4.0) |
| `2ffcfaeca6bcd57341b19c0f566e227b.json` | badcase | goodcase (bad=0.45, good=0.25) | badcase (bad=3.64, good=0.25) | badcase (bad=6.55, good=3.4) |
| `3151f0e14ae8cfa60f004c19524b1c53.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=1.11, good=0.5) | goodcase (bad=0.35, good=4.4826) |
| `32c4570cd5176ef38c7ebd2be698af48.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=0.68, good=0.5) | goodcase (bad=0.35, good=4.469) |
| `332bfa34d76eaa2033e2995f3286da82.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=1.11, good=0.5) | goodcase (bad=0.35, good=4.4802) |
| `35cbca0a8f4a69ce87bfc2d9650bb09c.json` | badcase | goodcase (bad=0.45, good=0.25) | badcase (bad=3.47, good=0.25) | badcase (bad=6.4058, good=0.25) |
| `3ca47c3c563ebbb1ee463756b9bc92f4.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=0.86, good=0.5) | goodcase (bad=0.35, good=4.4843) |
| `40287b3e5439e64cfaba87578c138931.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=0.68, good=0.5) | goodcase (bad=0.35, good=4.469) |
| `489fb4630e1b75cde0bbaff96c3c0c06.json` | badcase | goodcase (bad=0.45, good=0.25) | badcase (bad=3.76, good=0.25) | badcase (bad=8.04, good=3.9) |
| `4a0a69ca6327cf97f7041aac19356158.json` | badcase | goodcase (bad=0.0, good=0.5) | badcase (bad=0.86, good=0.5) | goodcase (bad=0.35, good=4.4836) |
| `4a6178560cfd56b94d8a7ce5767f11fe.json` | badcase | goodcase (bad=0.0, good=0.5) | badcase (bad=3.02, good=0.5) | badcase (bad=5.6093, good=0.5) |
| `5268073bfdf37fb4e35c0dbcdbab4202.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=2.7, good=0.5) | badcase (bad=5.21, good=3.4) |
| `53458e1dbfd6e39ef879399fac0fd97d-update.json` | badcase | goodcase (bad=0.45, good=0.25) | badcase (bad=2.26, good=0.25) | goodcase (bad=4.15, good=4.24) |
| `53458e1dbfd6e39ef879399fac0fd97d.json` | badcase | goodcase (bad=0.0, good=0.5) | badcase (bad=1.56, good=0.5) | goodcase (bad=3.3, good=4.5006) |
| `541f7730ced943209f9f5471ddc32000.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=0.68, good=0.5) | goodcase (bad=0.35, good=4.4722) |
| `5cba0dcf39751526846ad48a41b370b0.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=1.11, good=0.5) | goodcase (bad=3.0, good=4.4252) |
| `626ffb8a5eee4c8c644cdc20a6b82f3e-副本.json` | goodcase | goodcase (bad=0.45, good=0.25) | badcase (bad=3.47, good=0.25) | badcase (bad=4.0186, good=0.25) |
| `626ffb8a5eee4c8c644cdc20a6b82f3e.json` | badcase | goodcase (bad=0.45, good=0.25) | badcase (bad=3.47, good=0.25) | badcase (bad=4.0256, good=0.25) |
| `642b0d712a53438a83d2d08e646973f2.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=1.11, good=0.5) | goodcase (bad=0.35, good=4.1684) |
| `642e8f40d95d2a38c0548544b8602057.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=0.86, good=0.5) | goodcase (bad=0.0, good=4.148) |
| `671d7b6cb8fc70adec537ec5104ebd9a.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=3.22, good=0.5) | goodcase (bad=3.89, good=4.41) |
| `6dff8447fef60771e3c008e939f6e8a5.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=1.35, good=0.5) | goodcase (bad=0.24, good=4.336) |
| `75fdfd6cef5de10429bc17ee08d12671.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=0.86, good=0.5) | goodcase (bad=0.0, good=4.148) |
| `8142dd7bf7c063b609eeeaa84b355601.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=2.7, good=0.5) | badcase (bad=4.26, good=3.6) |
| `814bf629290d3bffc4d159fd72f01dd0.json` | goodcase | goodcase (bad=0.45, good=0.25) | badcase (bad=3.57, good=0.25) | badcase (bad=4.8, good=4.54) |
| `8171b7fc033b2e61191448f746b776a9.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=0.68, good=0.5) | goodcase (bad=0.35, good=4.469) |
| `825061b867bc9afd50a9cb9b7f00b02f.json` | goodcase | goodcase (bad=0.45, good=0.25) | badcase (bad=3.32, good=0.25) | badcase (bad=5.53, good=3.8) |
| `8335eb88438b4c85c7354be163e8a238.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=3.06, good=0.5) | badcase (bad=4.72, good=3.73) |
| `885ccada4a6cde66022efdbf5e143003.json` | badcase | goodcase (bad=0.0, good=0.5) | badcase (bad=1.75, good=0.5) | goodcase (bad=4.13, good=4.13) |
| `8e75cc06455515d5be52f20e80068cf2.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=0.68, good=0.5) | goodcase (bad=0.0, good=4.4713) |
| `91424fe7c896248c28a83a495f9d436b.json` | goodcase | goodcase (bad=0.45, good=0.25) | badcase (bad=3.57, good=0.25) | badcase (bad=5.29, good=4.0) |
| `91e548d8a5ec93845d058b3901d6ffa0.json` | badcase | goodcase (bad=0.45, good=0.25) | badcase (bad=3.47, good=0.25) | badcase (bad=6.0998, good=0.25) |
| `9596c3fb3096e8010e676e5e36f4c9d0.json` | badcase | goodcase (bad=0.0, good=0.5) | badcase (bad=0.86, good=0.5) | goodcase (bad=0.35, good=4.4697) |
| `962d3e2b785cefde6de631bdd8d4fd92.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=0.68, good=0.5) | goodcase (bad=0.35, good=4.469) |
| `98cb2fe2be979016f4b770f571a9f4ee.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=1.11, good=0.5) | goodcase (bad=0.59, good=4.4662) |
| `9d1e21658a0a908772bfb1f3e31c8456.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=1.29, good=0.5) | goodcase (bad=0.35, good=3.8752) |
| `a39bd72c1c05fcfd2dadbb61007f12d4.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=0.68, good=0.5) | goodcase (bad=0.35, good=4.4722) |
| `a553741020505b65949b923f755152a6.json` | goodcase | goodcase (bad=0.45, good=0.25) | badcase (bad=1.81, good=0.25) | goodcase (bad=2.33, good=3.877) |
| `a699424d557f03f0c60f668ef6409391.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=1.31, good=0.5) | goodcase (bad=0.35, good=3.9577) |
| `b024d8636b821f2e6760dab89bed9334.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=0.68, good=0.5) | goodcase (bad=0.35, good=4.469) |
| `b0878102726eec471e177a67ad32f2d5.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=0.75, good=0.5) | goodcase (bad=3.64, good=4.29) |
| `b0f76f80a8fa36ca4d38642d6a77999e.json` | goodcase | goodcase (bad=0.45, good=0.25) | badcase (bad=1.38, good=0.25) | goodcase (bad=0.8, good=3.8842) |
| `b75c4365c9e49667238bc9e3d7bc779d.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=1.11, good=0.5) | goodcase (bad=0.35, good=4.4823) |
| `bc0194e51161bf28201c780daccf36c6.json` | badcase | goodcase (bad=0.0, good=0.5) | badcase (bad=0.68, good=0.5) | goodcase (bad=0.83, good=3.9583) |
| `c5fdf5562fc823b2a57aab678dac491f.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=1.11, good=0.5) | goodcase (bad=0.59, good=4.4659) |
| `c8901c28294dcbd85443513875a2ca77.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=1.31, good=0.5) | badcase (bad=4.13, good=3.68) |
| `cc090bf913fbeab779172ad2d720f42e.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=1.29, good=0.5) | goodcase (bad=0.7, good=4.4435) |
| `d41de73cd94f085f04d4cde866b55c9d.json` | badcase | goodcase (bad=0.45, good=0.25) | badcase (bad=1.4, good=0.25) | goodcase (bad=3.33, good=3.88) |
| `d546fa569171b4b9eca83350a69ed7cb.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=0.68, good=0.5) | goodcase (bad=0.35, good=4.4722) |
| `dd301a0e584bf0c7ab344092416b9c5c.json` | badcase | goodcase (bad=0.45, good=0.25) | badcase (bad=3.47, good=0.25) | badcase (bad=4.3925, good=0.25) |
| `dd3150e0929f4c5e3325f12381dfb922.json` | badcase | goodcase (bad=0.0, good=0.5) | badcase (bad=1.47, good=0.5) | goodcase (bad=1.24, good=4.3486) |
| `dd3a4f6f77c8cb44ad7b892fb297c7e5.json` | badcase | goodcase (bad=0.0, good=0.5) | badcase (bad=2.97, good=0.5) | badcase (bad=2.61, good=0.5) |
| `e331687c86cad54da54527d5066b429f.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=0.68, good=0.5) | goodcase (bad=0.35, good=4.469) |
| `ef24be7ae66f56916ff306cddd269f71.json` | badcase | goodcase (bad=0.0, good=0.5) | badcase (bad=2.77, good=0.5) | badcase (bad=3.2909, good=0.5) |
| `f0c5c6877de7bc51ff9fc455a1f6997c.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=1.43, good=0.5) | goodcase (bad=0.35, good=4.4553) |
| `f3c59071bba65a9d3da6d8330a039b05.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=0.68, good=0.5) | goodcase (bad=0.35, good=4.469) |
| `f6236cea86a888d986b6c4a751723dd3.json` | badcase | goodcase (bad=0.45, good=0.25) | badcase (bad=3.47, good=0.25) | badcase (bad=6.161, good=0.25) |
| `f6ebb22c2c920bf5ea27ab04e8a8962f.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=3.3, good=0.5) | badcase (bad=2.61, good=0.5) |
| `fbf91ecd7fdf951f25664bdd7ee607f5.json` | badcase | goodcase (bad=0.0, good=0.5) | badcase (bad=3.22, good=0.5) | badcase (bad=2.61, good=0.5) |
| `ff345d8ef7300db31b69cb579a2cfb57.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=1.35, good=0.5) | goodcase (bad=1.19, good=4.33) |
| `ff8cbab7dfb7d56901186c800a0eceed.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=2.68, good=0.5) | goodcase (bad=3.89, good=4.19) |
| `一句话3-6.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=1.76, good=0.5) | badcase (bad=4.0, good=3.64) |
| `一句话训练场景.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=3.4, good=0.5) | badcase (bad=4.43, good=3.48) |
| `交大卓越中心签约仪式策划报告2.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=1.35, good=0.5) | goodcase (bad=0.49, good=3.38) |
| `策划报告-King Abdul Aziz University7.json` | goodcase | goodcase (bad=0.45, good=0.25) | badcase (bad=5.4, good=0.25) | badcase (bad=5.29, good=3.21) |
| `策划报告-M3T1A3688N1218970094518456402.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=1.67, good=0.5) | goodcase (bad=0.24, good=4.5247) |
| `策划报告-M3T1A3688N1218971737170698461-6.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=3.22, good=0.5) | goodcase (bad=3.89, good=4.49) |
| `策划报告-Ministry Of Higher Education. ARE-V2-7.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=1.77, good=0.5) | badcase (bad=4.14, good=3.34) |
| `策划报告-上海交大卓越中心5.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=1.31, good=0.5) | badcase (bad=4.13, good=3.68) |
| `策划报告-巴拿马7.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=4.39, good=0.5) | badcase (bad=4.37, good=3.46) |
| `策划报告-朱拉隆功7.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=3.07, good=0.5) | goodcase (bad=4.36, good=4.41) |
| `策划报告（张校长）10.json` | goodcase | goodcase (bad=0.0, good=0.5) | badcase (bad=2.31, good=0.5) | goodcase (bad=3.89, good=4.49) |
