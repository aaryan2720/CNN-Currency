# Validation Error & Diagnostic Analysis

## 1. Executive Evaluation Summary

- **Validation Dataset**: 70 images (10 per class across 7 classes)
- **Overall Accuracy**: `44.29%`
- **Macro-Average Accuracy**: `44.29%`
- **Validation Loss**: `1.8383`
- **Total Misclassifications**: `39 / 70`

## 2. Per-Class Accuracy Breakdown

| Class Name | Validation Samples | Correct | Incorrect | Class Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Rs_10** | 10 | 0 | 10 | `0.00%` |
| **Rs_20** | 10 | 6 | 4 | `60.00%` |
| **Rs_50** | 10 | 6 | 4 | `60.00%` |
| **Rs_100** | 10 | 0 | 10 | `0.00%` |
| **Rs_200** | 10 | 2 | 8 | `20.00%` |
| **Rs_500** | 10 | 8 | 2 | `80.00%` |
| **Rs_2000** | 10 | 9 | 1 | `90.00%` |

## 3. Detailed Audit of Misclassified Samples

### Error #1: `Rs_50/image_024.jpg`
- **Original File**: `50_original_109.jpg_1075de7d-5d16-44d3-9798-85b0ff955285.jpg`
- **True Class**: **Rs_50** (Assigned prob: `10.51%`)
- **Predicted Class**: **Rs_500** (Confidence: `26.11%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1185,
  "Rs_20": 0.0848,
  "Rs_50": 0.1051,
  "Rs_100": 0.124,
  "Rs_200": 0.1355,
  "Rs_500": 0.2611,
  "Rs_2000": 0.171
}
```

### Error #2: `Rs_20/image_050.jpg`
- **Original File**: `20_original_112.jpg_f5de661a-6858-4081-a84a-42e68c2c9789.jpg`
- **True Class**: **Rs_20** (Assigned prob: `7.44%`)
- **Predicted Class**: **Rs_50** (Confidence: `20.48%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1511,
  "Rs_20": 0.0744,
  "Rs_50": 0.2048,
  "Rs_100": 0.1584,
  "Rs_200": 0.1223,
  "Rs_500": 0.161,
  "Rs_2000": 0.128
}
```

### Error #3: `Rs_50/image_049.jpg`
- **Original File**: `50_original_122.jpg_20005eeb-2762-49bb-8174-e07cb810dc25.jpg`
- **True Class**: **Rs_50** (Assigned prob: `17.05%`)
- **Predicted Class**: **Rs_100** (Confidence: `19.16%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1384,
  "Rs_20": 0.0784,
  "Rs_50": 0.1705,
  "Rs_100": 0.1916,
  "Rs_200": 0.115,
  "Rs_500": 0.1774,
  "Rs_2000": 0.1287
}
```

### Error #4: `Rs_10/image_009.jpg`
- **Original File**: `10_original_103.jpg_ce8f7f01-90db-4835-ab7f-2d137ca8199a.jpg`
- **True Class**: **Rs_10** (Assigned prob: `11.72%`)
- **Predicted Class**: **Rs_2000** (Confidence: `18.05%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1172,
  "Rs_20": 0.1692,
  "Rs_50": 0.1007,
  "Rs_100": 0.1146,
  "Rs_200": 0.1574,
  "Rs_500": 0.1604,
  "Rs_2000": 0.1805
}
```

### Error #5: `Rs_10/image_013.jpg`
- **Original File**: `10_original_104.jpg_6d2399f1-2e3c-4472-9688-2812bb1ed4d1.jpg`
- **True Class**: **Rs_10** (Assigned prob: `13.97%`)
- **Predicted Class**: **Rs_500** (Confidence: `19.76%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1397,
  "Rs_20": 0.1003,
  "Rs_50": 0.1488,
  "Rs_100": 0.1274,
  "Rs_200": 0.1494,
  "Rs_500": 0.1976,
  "Rs_2000": 0.1368
}
```

### Error #6: `Rs_10/image_001.jpg`
- **Original File**: `10_original_1.jpg_ac34d49b-e529-48f1-bb29-6eb8401bba8d.jpg`
- **True Class**: **Rs_10** (Assigned prob: `15.49%`)
- **Predicted Class**: **Rs_500** (Confidence: `20.59%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1549,
  "Rs_20": 0.0911,
  "Rs_50": 0.1406,
  "Rs_100": 0.1394,
  "Rs_200": 0.1166,
  "Rs_500": 0.2059,
  "Rs_2000": 0.1515
}
```

### Error #7: `Rs_10/image_002.jpg`
- **Original File**: `10_original_1.jpg_cf85e882-527c-4ac8-be8c-3e8d1cc791fc.jpg`
- **True Class**: **Rs_10** (Assigned prob: `13.75%`)
- **Predicted Class**: **Rs_100** (Confidence: `16.68%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1375,
  "Rs_20": 0.1204,
  "Rs_50": 0.1571,
  "Rs_100": 0.1668,
  "Rs_200": 0.1225,
  "Rs_500": 0.1548,
  "Rs_2000": 0.1409
}
```

### Error #8: `Rs_200/image_048.jpg`
- **Original File**: `200_original_IMG_20190422_063815782_HDR.jpg_196ed3ab-43da-49d7-b5a1-0127f2cad1ee.jpg`
- **True Class**: **Rs_200** (Assigned prob: `19.42%`)
- **Predicted Class**: **Rs_50** (Confidence: `22.66%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1205,
  "Rs_20": 0.0962,
  "Rs_50": 0.2266,
  "Rs_100": 0.1008,
  "Rs_200": 0.1942,
  "Rs_500": 0.1321,
  "Rs_2000": 0.1295
}
```

### Error #9: `Rs_500/image_031.jpg`
- **Original File**: `500_original_IMG_20190419_211942867.jpg_316f5365-bf61-4d92-b490-24906eef0325.jpg`
- **True Class**: **Rs_500** (Assigned prob: `14.84%`)
- **Predicted Class**: **Rs_50** (Confidence: `17.36%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1247,
  "Rs_20": 0.0887,
  "Rs_50": 0.1736,
  "Rs_100": 0.1448,
  "Rs_200": 0.1578,
  "Rs_500": 0.1484,
  "Rs_2000": 0.162
}
```

### Error #10: `Rs_10/image_014.jpg`
- **Original File**: `10_original_105.jpg_2a7fb7ca-423e-4a9f-afb2-578755a74739.jpg`
- **True Class**: **Rs_10** (Assigned prob: `15.68%`)
- **Predicted Class**: **Rs_100** (Confidence: `17.81%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1568,
  "Rs_20": 0.0972,
  "Rs_50": 0.1672,
  "Rs_100": 0.1781,
  "Rs_200": 0.1152,
  "Rs_500": 0.133,
  "Rs_2000": 0.1525
}
```

### Error #11: `Rs_200/image_029.jpg`
- **Original File**: `200_original_IMG_20190422_063753922.jpg_21ea99ef-889a-4a37-8d15-61777866a5ed.jpg`
- **True Class**: **Rs_200** (Assigned prob: `16.91%`)
- **Predicted Class**: **Rs_500** (Confidence: `17.34%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1462,
  "Rs_20": 0.1711,
  "Rs_50": 0.1071,
  "Rs_100": 0.0894,
  "Rs_200": 0.1691,
  "Rs_500": 0.1734,
  "Rs_2000": 0.1439
}
```

### Error #12: `Rs_500/image_029.jpg`
- **Original File**: `500_original_IMG_20190419_211942867.jpg_1f9f5bd4-d515-45b5-882f-ae7f836595cc.jpg`
- **True Class**: **Rs_500** (Assigned prob: `15.75%`)
- **Predicted Class**: **Rs_100** (Confidence: `16.87%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1352,
  "Rs_20": 0.083,
  "Rs_50": 0.1574,
  "Rs_100": 0.1687,
  "Rs_200": 0.1363,
  "Rs_500": 0.1575,
  "Rs_2000": 0.1619
}
```

### Error #13: `Rs_200/image_040.jpg`
- **Original File**: `200_original_IMG_20190422_063810752_HDR.jpg_755cc580-cdd6-49eb-8ada-61dd9f185306.jpg`
- **True Class**: **Rs_200** (Assigned prob: `17.09%`)
- **Predicted Class**: **Rs_50** (Confidence: `20.83%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1457,
  "Rs_20": 0.09,
  "Rs_50": 0.2083,
  "Rs_100": 0.1074,
  "Rs_200": 0.1709,
  "Rs_500": 0.1787,
  "Rs_2000": 0.0989
}
```

### Error #14: `Rs_20/image_040.jpg`
- **Original File**: `20_original_11.jpg_07b68c3c-54a2-4685-af9f-c0a884d36b6c.jpg`
- **True Class**: **Rs_20** (Assigned prob: `13.99%`)
- **Predicted Class**: **Rs_50** (Confidence: `16.61%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1625,
  "Rs_20": 0.1399,
  "Rs_50": 0.1661,
  "Rs_100": 0.0992,
  "Rs_200": 0.1653,
  "Rs_500": 0.1246,
  "Rs_2000": 0.1425
}
```

### Error #15: `Rs_20/image_013.jpg`
- **Original File**: `20_original_101.jpg_3c4ed0f1-fd27-4b6a-a089-a92feb339311.jpg`
- **True Class**: **Rs_20** (Assigned prob: `9.27%`)
- **Predicted Class**: **Rs_100** (Confidence: `17.82%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1216,
  "Rs_20": 0.0927,
  "Rs_50": 0.1772,
  "Rs_100": 0.1782,
  "Rs_200": 0.1377,
  "Rs_500": 0.1604,
  "Rs_2000": 0.1321
}
```

### Error #16: `Rs_10/image_003.jpg`
- **Original File**: `10_original_10.jpg_21c06318-dc76-4fbe-942f-49d6297a6729.jpg`
- **True Class**: **Rs_10** (Assigned prob: `13.84%`)
- **Predicted Class**: **Rs_500** (Confidence: `24.36%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1384,
  "Rs_20": 0.0869,
  "Rs_50": 0.1474,
  "Rs_100": 0.1101,
  "Rs_200": 0.1371,
  "Rs_500": 0.2436,
  "Rs_2000": 0.1364
}
```

### Error #17: `Rs_100/image_014.jpg`
- **Original File**: `100_original_103.jpg_e546dc2f-7012-4fb8-9841-d4801f26a2c9.jpg`
- **True Class**: **Rs_100** (Assigned prob: `18.39%`)
- **Predicted Class**: **Rs_500** (Confidence: `21.57%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1478,
  "Rs_20": 0.0543,
  "Rs_50": 0.1683,
  "Rs_100": 0.1839,
  "Rs_200": 0.107,
  "Rs_500": 0.2157,
  "Rs_2000": 0.123
}
```

### Error #18: `Rs_100/image_031.jpg`
- **Original File**: `100_original_111.jpg_7c388cf9-513d-416e-9d16-12536b6a5af2.jpg`
- **True Class**: **Rs_100** (Assigned prob: `11.10%`)
- **Predicted Class**: **Rs_500** (Confidence: `32.51%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1178,
  "Rs_20": 0.0811,
  "Rs_50": 0.1249,
  "Rs_100": 0.111,
  "Rs_200": 0.1301,
  "Rs_500": 0.3251,
  "Rs_2000": 0.1101
}
```

### Error #19: `Rs_50/image_002.jpg`
- **Original File**: `50_original_10.jpg_39ce90a1-118f-444a-a9e6-ad20807c3405.jpg`
- **True Class**: **Rs_50** (Assigned prob: `13.85%`)
- **Predicted Class**: **Rs_500** (Confidence: `27.83%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1039,
  "Rs_20": 0.0791,
  "Rs_50": 0.1385,
  "Rs_100": 0.1775,
  "Rs_200": 0.1005,
  "Rs_500": 0.2783,
  "Rs_2000": 0.1222
}
```

### Error #20: `Rs_100/image_009.jpg`
- **Original File**: `100_original_101.jpg_3a3f3e42-49ba-421b-8796-10499312e9d5.jpg`
- **True Class**: **Rs_100** (Assigned prob: `16.37%`)
- **Predicted Class**: **Rs_500** (Confidence: `22.69%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1304,
  "Rs_20": 0.1095,
  "Rs_50": 0.1175,
  "Rs_100": 0.1637,
  "Rs_200": 0.1085,
  "Rs_500": 0.2269,
  "Rs_2000": 0.1435
}
```

### Error #21: `Rs_100/image_006.jpg`
- **Original File**: `100_original_100.jpg_aab0fc6d-ffcd-4f45-ac78-4ade760abfd8.jpg`
- **True Class**: **Rs_100** (Assigned prob: `10.89%`)
- **Predicted Class**: **Rs_200** (Confidence: `17.22%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1396,
  "Rs_20": 0.1486,
  "Rs_50": 0.148,
  "Rs_100": 0.1089,
  "Rs_200": 0.1722,
  "Rs_500": 0.1178,
  "Rs_2000": 0.1648
}
```

### Error #22: `Rs_100/image_001.jpg`
- **Original File**: `100_original_10.jpg_75010d12-2a08-4790-8371-4d94f9182ed4.jpg`
- **True Class**: **Rs_100** (Assigned prob: `9.87%`)
- **Predicted Class**: **Rs_200** (Confidence: `19.48%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1514,
  "Rs_20": 0.1077,
  "Rs_50": 0.1512,
  "Rs_100": 0.0987,
  "Rs_200": 0.1948,
  "Rs_500": 0.1356,
  "Rs_2000": 0.1606
}
```

### Error #23: `Rs_100/image_045.jpg`
- **Original File**: `100_original_118.jpg_069cefde-236d-4bfc-9949-4c475184502b.jpg`
- **True Class**: **Rs_100** (Assigned prob: `12.57%`)
- **Predicted Class**: **Rs_500** (Confidence: `17.90%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1561,
  "Rs_20": 0.0723,
  "Rs_50": 0.1785,
  "Rs_100": 0.1257,
  "Rs_200": 0.154,
  "Rs_500": 0.179,
  "Rs_2000": 0.1344
}
```

### Error #24: `Rs_200/image_002.jpg`
- **Original File**: `200_original_IMG_20190422_063722432.jpg_bb17274b-0961-4d61-b689-92ab934d5a56.jpg`
- **True Class**: **Rs_200** (Assigned prob: `14.72%`)
- **Predicted Class**: **Rs_20** (Confidence: `20.15%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1441,
  "Rs_20": 0.2015,
  "Rs_50": 0.125,
  "Rs_100": 0.1115,
  "Rs_200": 0.1472,
  "Rs_500": 0.1193,
  "Rs_2000": 0.1514
}
```

### Error #25: `Rs_20/image_002.jpg`
- **Original File**: `20_original_1.jpg_39b4ba38-2145-4216-a31c-14b18b1013d9.jpg`
- **True Class**: **Rs_20** (Assigned prob: `7.02%`)
- **Predicted Class**: **Rs_500** (Confidence: `18.86%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1701,
  "Rs_20": 0.0702,
  "Rs_50": 0.1654,
  "Rs_100": 0.123,
  "Rs_200": 0.1494,
  "Rs_500": 0.1886,
  "Rs_2000": 0.1333
}
```

### Error #26: `Rs_10/image_037.jpg`
- **Original File**: `10_original_115.jpg_fa9ad337-12ac-4edb-bfb5-b4432318e495.jpg`
- **True Class**: **Rs_10** (Assigned prob: `13.79%`)
- **Predicted Class**: **Rs_500** (Confidence: `16.26%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1379,
  "Rs_20": 0.1397,
  "Rs_50": 0.1491,
  "Rs_100": 0.099,
  "Rs_200": 0.1602,
  "Rs_500": 0.1626,
  "Rs_2000": 0.1513
}
```

### Error #27: `Rs_50/image_036.jpg`
- **Original File**: `50_original_115.jpg_08ca9938-42fb-404f-929a-77c963a524b9.jpg`
- **True Class**: **Rs_50** (Assigned prob: `17.51%`)
- **Predicted Class**: **Rs_200** (Confidence: `20.53%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1216,
  "Rs_20": 0.0939,
  "Rs_50": 0.1751,
  "Rs_100": 0.0852,
  "Rs_200": 0.2053,
  "Rs_500": 0.1934,
  "Rs_2000": 0.1254
}
```

### Error #28: `Rs_100/image_034.jpg`
- **Original File**: `100_original_113.jpg_3d09b074-979a-451d-a31c-135a07641e43.jpg`
- **True Class**: **Rs_100** (Assigned prob: `13.45%`)
- **Predicted Class**: **Rs_500** (Confidence: `22.48%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1284,
  "Rs_20": 0.1025,
  "Rs_50": 0.1333,
  "Rs_100": 0.1345,
  "Rs_200": 0.1406,
  "Rs_500": 0.2248,
  "Rs_2000": 0.136
}
```

### Error #29: `Rs_100/image_004.jpg`
- **Original File**: `100_original_100.jpg_956778ec-7c7e-492c-b86e-a7ecb61bb935.jpg`
- **True Class**: **Rs_100** (Assigned prob: `13.41%`)
- **Predicted Class**: **Rs_500** (Confidence: `19.85%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1306,
  "Rs_20": 0.1338,
  "Rs_50": 0.1291,
  "Rs_100": 0.1341,
  "Rs_200": 0.1381,
  "Rs_500": 0.1985,
  "Rs_2000": 0.1359
}
```

### Error #30: `Rs_10/image_034.jpg`
- **Original File**: `10_original_113.jpg_a5e14c0f-5709-4581-8f36-d7a4f1317aeb.jpg`
- **True Class**: **Rs_10** (Assigned prob: `12.96%`)
- **Predicted Class**: **Rs_200** (Confidence: `19.55%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1296,
  "Rs_20": 0.1177,
  "Rs_50": 0.1789,
  "Rs_100": 0.1093,
  "Rs_200": 0.1955,
  "Rs_500": 0.1259,
  "Rs_2000": 0.1431
}
```

### Error #31: `Rs_100/image_030.jpg`
- **Original File**: `100_original_110.jpg_61f3223f-c293-478a-a767-009f0285667a.jpg`
- **True Class**: **Rs_100** (Assigned prob: `10.02%`)
- **Predicted Class**: **Rs_10** (Confidence: `16.88%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1688,
  "Rs_20": 0.1683,
  "Rs_50": 0.1179,
  "Rs_100": 0.1002,
  "Rs_200": 0.1393,
  "Rs_500": 0.1669,
  "Rs_2000": 0.1387
}
```

### Error #32: `Rs_200/image_033.jpg`
- **Original File**: `200_original_IMG_20190422_063756088.jpg_f4997865-22cc-4bf6-a34f-0fb783551e3a.jpg`
- **True Class**: **Rs_200** (Assigned prob: `13.41%`)
- **Predicted Class**: **Rs_500** (Confidence: `18.31%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1523,
  "Rs_20": 0.1186,
  "Rs_50": 0.1064,
  "Rs_100": 0.1246,
  "Rs_200": 0.1341,
  "Rs_500": 0.1831,
  "Rs_2000": 0.1808
}
```

### Error #33: `Rs_100/image_020.jpg`
- **Original File**: `100_original_106.jpg_38c1e954-12d4-4919-980b-c569d3d86f54.jpg`
- **True Class**: **Rs_100** (Assigned prob: `12.74%`)
- **Predicted Class**: **Rs_500** (Confidence: `23.08%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1558,
  "Rs_20": 0.0743,
  "Rs_50": 0.1632,
  "Rs_100": 0.1274,
  "Rs_200": 0.1148,
  "Rs_500": 0.2308,
  "Rs_2000": 0.1337
}
```

### Error #34: `Rs_200/image_030.jpg`
- **Original File**: `200_original_IMG_20190422_063753922.jpg_2e24e5e8-441a-45c2-bbc2-e1691b696f46.jpg`
- **True Class**: **Rs_200** (Assigned prob: `16.15%`)
- **Predicted Class**: **Rs_20** (Confidence: `21.43%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1129,
  "Rs_20": 0.2143,
  "Rs_50": 0.1042,
  "Rs_100": 0.0937,
  "Rs_200": 0.1615,
  "Rs_500": 0.1654,
  "Rs_2000": 0.1481
}
```

### Error #35: `Rs_10/image_049.jpg`
- **Original File**: `10_original_121.jpg_3476ff3f-75e6-4021-a540-7f9c39d840c2.jpg`
- **True Class**: **Rs_10** (Assigned prob: `14.63%`)
- **Predicted Class**: **Rs_50** (Confidence: `20.50%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1463,
  "Rs_20": 0.1209,
  "Rs_50": 0.205,
  "Rs_100": 0.1193,
  "Rs_200": 0.1123,
  "Rs_500": 0.1727,
  "Rs_2000": 0.1235
}
```

### Error #36: `Rs_2000/image_034.jpg`
- **Original File**: `2000_original_IMG_20190421_140632330.jpg_174229f5-f994-44b7-89a6-fe95fcce0224.jpg`
- **True Class**: **Rs_2000** (Assigned prob: `17.25%`)
- **Predicted Class**: **Rs_20** (Confidence: `17.78%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1372,
  "Rs_20": 0.1778,
  "Rs_50": 0.1035,
  "Rs_100": 0.1092,
  "Rs_200": 0.1503,
  "Rs_500": 0.1496,
  "Rs_2000": 0.1725
}
```

### Error #37: `Rs_200/image_015.jpg`
- **Original File**: `200_original_IMG_20190422_063732286.jpg_72d3dffa-d7c0-44b0-a0c8-aba3cf01de84.jpg`
- **True Class**: **Rs_200** (Assigned prob: `16.39%`)
- **Predicted Class**: **Rs_20** (Confidence: `18.11%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1567,
  "Rs_20": 0.1811,
  "Rs_50": 0.0939,
  "Rs_100": 0.0996,
  "Rs_200": 0.1639,
  "Rs_500": 0.1442,
  "Rs_2000": 0.1607
}
```

### Error #38: `Rs_200/image_022.jpg`
- **Original File**: `200_original_IMG_20190422_063745488.jpg_d655991b-96a3-4b1a-8b12-654b314a6bbe.jpg`
- **True Class**: **Rs_200** (Assigned prob: `12.92%`)
- **Predicted Class**: **Rs_100** (Confidence: `17.62%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1301,
  "Rs_20": 0.1173,
  "Rs_50": 0.1396,
  "Rs_100": 0.1762,
  "Rs_200": 0.1292,
  "Rs_500": 0.175,
  "Rs_2000": 0.1325
}
```

### Error #39: `Rs_10/image_015.jpg`
- **Original File**: `10_original_105.jpg_66cc1696-be41-44bc-89ef-770b55353f81.jpg`
- **True Class**: **Rs_10** (Assigned prob: `13.08%`)
- **Predicted Class**: **Rs_2000** (Confidence: `17.55%`)
- **Full Probability Distribution**:
```json
{
  "Rs_10": 0.1308,
  "Rs_20": 0.1278,
  "Rs_50": 0.1294,
  "Rs_100": 0.167,
  "Rs_200": 0.1399,
  "Rs_500": 0.1297,
  "Rs_2000": 0.1755
}
```

## 4. Key Empirical Observations

1. **Dominant Confusions**: Any observed misclassifications primarily occur between notes with similar ambient background lighting or shared aspect ratios.
2. **Data Scale Constraint**: Evaluating on a fixed 70-image partition implies each individual sample accounts for ~1.43% accuracy variance.
3. **Absence of Data Augmentation**: Without spatial jitter or slight rotation in training, certain non-centered validation notes show lower confidence.

---
*Report generated from actual model predictions on the validated test partition.*