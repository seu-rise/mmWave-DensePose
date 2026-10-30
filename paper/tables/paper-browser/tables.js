globalThis.PAPER_TABLE_DATA = {
  "tables": {
    "1": {
      "id": "1",
      "title": "Data streams",
      "reference": {
        "columns": [
          "Data stream",
          "Source",
          "Main content",
          "Role in this paper"
        ],
        "rows": [
          [
            "Radar point cloud",
            "mmWave radar",
            "Position, Doppler informatio n , and SNR",
            "Network input"
          ],
          [
            "Skeleton",
            "Kinect V2",
            "3D skeletal joints",
            "Ground-truth annotation"
          ],
          [
            "RGB image",
            "Kinect V2",
            "Color image",
            "Reference"
          ]
        ]
      },
      "measured": {
        "columns": [
          "Data stream",
          "Source",
          "Main content",
          "Role in this paper"
        ],
        "rows": [
          [
            "Radar point cloud",
            null,
            null,
            null
          ],
          [
            "Skeleton",
            null,
            null,
            null
          ],
          [
            "RGB image",
            null,
            null,
            null
          ]
        ]
      },
      "provenance": [],
      "reference_source": "paper/reference/table_01.csv",
      "reference_sha256": "c3495b4003959316a25ba3338be0f43196bc1953c5591f8332b27d6c737fd952"
    },
    "2": {
      "id": "2",
      "title": "Dataset statistics",
      "reference": {
        "columns": [
          "Item",
          "Value"
        ],
        "rows": [
          [
            "Number of subjects",
            "101"
          ],
          [
            "Motion coverage",
            "19 predefined motions + free motion"
          ],
          [
            "Number of sequences",
            "141"
          ],
          [
            "Mean frames per sequence",
            "4043.43"
          ],
          [
            "Valid synchronized frames",
            "570 124"
          ],
          [
            "Mean valid radar points per frame",
            "164"
          ],
          [
            "Radar point attributes",
            "X , Y , Z ,Doppler i nforamtion , and SNR"
          ],
          [
            "Skeleton annotation",
            "19 × 3"
          ]
        ]
      },
      "measured": {
        "columns": [
          "Item",
          "Value"
        ],
        "rows": [
          [
            "Number of subjects",
            null
          ],
          [
            "Motion coverage",
            null
          ],
          [
            "Number of sequences",
            null
          ],
          [
            "Mean frames per sequence",
            null
          ],
          [
            "Valid synchronized frames",
            null
          ],
          [
            "Mean valid radar points per frame",
            null
          ],
          [
            "Radar point attributes",
            null
          ],
          [
            "Skeleton annotation",
            null
          ]
        ]
      },
      "provenance": [],
      "reference_source": "paper/reference/table_02.csv",
      "reference_sha256": "4d0ba893a448659a92fc8421a92840fa97c91a737834e9a91c5e7694e5d7b8e1"
    },
    "3": {
      "id": "3",
      "title": "Related datasets",
      "reference": {
        "columns": [
          "Dataset",
          "Hardware device",
          "Array configuration",
          "Number of subjects",
          "Number of frames",
          "Motion"
        ],
        "rows": [
          [
            "mmPose (2020) [13]",
            "2 × TI AWR1642BOOST",
            "2T4R",
            "2",
            "∼40 000",
            "4 motions"
          ],
          [
            "MARS (2021) [58]",
            "TI IWR1443BOOST",
            "3T4R",
            "4",
            "40 000",
            "10 motions"
          ],
          [
            "mRI (2022) [43]",
            "TI IWR1443BOOST",
            "3T4R",
            "20",
            "∼160 000",
            "12 motions"
          ],
          [
            "mmPose-NLP (2023) [17]",
            "2 × TI AWR1843",
            "3T4R",
            "2",
            "∼16 000",
            "4 motions"
          ],
          [
            "HuPR (2023) [25]",
            "2 × TI IWR1843BOOST",
            "3T4R",
            "6",
            "141 000",
            "3 motions"
          ],
          [
            "MiliPoint (2023) [42]",
            "TI IWR1843",
            "3T4R",
            "11",
            "545 000",
            "49 motions"
          ],
          [
            "mmPose-FK (2024) [52]",
            "TI IWR6843ISK-ODS",
            "3T4R",
            "30",
            "∼215 000",
            "2 motions a"
          ],
          [
            "mmPrivPose3D (2025) [59]",
            "TI IWR6843AOPEVM",
            "3T4R",
            "15",
            "—",
            "3 motions"
          ],
          [
            "mmWave-DensePose (ours)",
            "Customized MIMO radar",
            "12T16R",
            "101",
            "570 000",
            "19 motions + free b"
          ]
        ]
      },
      "measured": {
        "columns": [
          "Dataset",
          "Hardware device",
          "Array configuration",
          "Number of subjects",
          "Number of frames",
          "Motion"
        ],
        "rows": [
          [
            "mmPose (2020) [13]",
            null,
            null,
            null,
            null,
            null
          ],
          [
            "MARS (2021) [58]",
            null,
            null,
            null,
            null,
            null
          ],
          [
            "mRI (2022) [43]",
            null,
            null,
            null,
            null,
            null
          ],
          [
            "mmPose-NLP (2023) [17]",
            null,
            null,
            null,
            null,
            null
          ],
          [
            "HuPR (2023) [25]",
            null,
            null,
            null,
            null,
            null
          ],
          [
            "MiliPoint (2023) [42]",
            null,
            null,
            null,
            null,
            null
          ],
          [
            "mmPose-FK (2024) [52]",
            null,
            null,
            null,
            null,
            null
          ],
          [
            "mmPrivPose3D (2025) [59]",
            null,
            null,
            null,
            null,
            null
          ],
          [
            "mmWave-DensePose (ours)",
            null,
            null,
            null,
            null,
            null
          ]
        ]
      },
      "provenance": [],
      "reference_source": "paper/reference/table_03.csv",
      "reference_sha256": "ec120c74be6ff4a5371c3d7fdefc1953fdde8a8d9cf3b60ef3178292df6cec24"
    },
    "4": {
      "id": "4",
      "title": "Overall performance",
      "reference": {
        "columns": [
          "Method",
          "MLE (cm)",
          "PA-MPJPE (cm)",
          "PCK@15cm (%)",
          "AUC@15cm (%)"
        ],
        "rows": [
          [
            "DGCNN (2019) [62]",
            7.3,
            5.29,
            92.33,
            55.5
          ],
          [
            "PointTransformer (2021) [60]",
            7.17,
            5.21,
            92.85,
            55.51
          ],
          [
            "PointMLP (2022) [61]",
            5.6,
            4.07,
            96.45,
            64.39
          ],
          [
            "mmPose-FK (2024) [52] a",
            5.57,
            4.21,
            97.07,
            63.92
          ],
          [
            "Dual-Path CNN–BiLSTM (2025) [21]",
            8.19,
            5.46,
            90.79,
            49.91
          ],
          [
            "PntPoseAT (2025) [53]",
            5.24,
            3.98,
            97.59,
            65.96
          ],
          [
            "PoseGraphNet (2026) [51]",
            6.7,
            5.17,
            94.23,
            57.77
          ],
          [
            "Global–Local Network (2026) [30]",
            9.18,
            6.09,
            87.26,
            44.48
          ],
          [
            "PAST-Net (ours)",
            4.08,
            3.17,
            98.69,
            73.35
          ]
        ]
      },
      "measured": {
        "columns": [
          "Method",
          "MLE (cm)",
          "PA-MPJPE (cm)",
          "PCK@15cm (%)",
          "AUC@15cm (%)"
        ],
        "rows": [
          [
            "DGCNN (2019) [62]",
            null,
            null,
            null,
            null
          ],
          [
            "PointTransformer (2021) [60]",
            null,
            null,
            null,
            null
          ],
          [
            "PointMLP (2022) [61]",
            null,
            null,
            null,
            null
          ],
          [
            "mmPose-FK (2024) [52] a",
            null,
            null,
            null,
            null
          ],
          [
            "Dual-Path CNN–BiLSTM (2025) [21]",
            null,
            null,
            null,
            null
          ],
          [
            "PntPoseAT (2025) [53]",
            null,
            null,
            null,
            null
          ],
          [
            "PoseGraphNet (2026) [51]",
            null,
            null,
            null,
            null
          ],
          [
            "Global–Local Network (2026) [30]",
            null,
            null,
            null,
            null
          ],
          [
            "PAST-Net (ours)",
            null,
            null,
            null,
            null
          ]
        ]
      },
      "provenance": [],
      "reference_source": "paper/reference/table_04.csv",
      "reference_sha256": "923ba1d982e0561a916b64b7623672aad611c162d204bfb437cf800acb56e068"
    },
    "5": {
      "id": "5",
      "title": "Per-joint MLE",
      "reference": {
        "columns": [
          "Method",
          "J0 (cm)",
          "J1 (cm)",
          "J2 (cm)",
          "J3 (cm)",
          "J4 (cm)",
          "J5 (cm)",
          "J6 (cm)",
          "J7 (cm)",
          "J8 (cm)",
          "J9 (cm)",
          "J10 (cm)",
          "J11 (cm)",
          "J12 (cm)",
          "J13 (cm)",
          "J14 (cm)",
          "J15 (cm)",
          "J16 (cm)",
          "J17 (cm)",
          "J18 (cm)",
          "Average (cm)"
        ],
        "rows": [
          [
            "DGCNN (2019) [62]",
            5.55,
            4.25,
            4.48,
            4.76,
            5.05,
            7.69,
            11.14,
            4.84,
            7.26,
            10.97,
            5.61,
            8.32,
            9.51,
            10.88,
            5.58,
            8.23,
            9.36,
            10.89,
            4.25,
            7.3
          ],
          [
            "PointTransformer (2021) [60]",
            5.63,
            4.39,
            4.9,
            5.81,
            5.19,
            7.13,
            10.83,
            5.07,
            6.99,
            10.73,
            5.69,
            7.51,
            9.03,
            10.35,
            5.69,
            7.49,
            8.86,
            10.24,
            4.61,
            7.17
          ],
          [
            "PointMLP (2022) [61]",
            4.39,
            3.49,
            3.78,
            4.01,
            4.1,
            5.61,
            7.96,
            3.9,
            5.4,
            7.84,
            4.44,
            5.86,
            7.32,
            8.73,
            4.45,
            5.8,
            7.08,
            8.63,
            3.59,
            5.6
          ],
          [
            "mmPose-FK (2024) [52] a,b",
            3.8,
            3.91,
            4.45,
            4.38,
            4.34,
            5.89,
            9.02,
            4.19,
            6.06,
            9.0,
            4.15,
            5.58,
            6.67,
            null,
            4.15,
            6.99,
            6.67,
            null,
            5.48,
            5.57
          ],
          [
            "DP-CNN–BiLSTM (2025) [21] a,c",
            6.01,
            5.74,
            6.5,
            7.33,
            7.03,
            10.29,
            18.25,
            6.19,
            7.19,
            10.25,
            6.18,
            7.84,
            9.75,
            11.17,
            5.93,
            6.66,
            7.72,
            9.32,
            6.23,
            8.19
          ],
          [
            "PntPoseAT (2025) [53] a",
            3.91,
            3.14,
            3.49,
            3.91,
            3.91,
            5.73,
            8.2,
            3.78,
            5.75,
            8.28,
            3.97,
            5.42,
            6.05,
            7.65,
            3.98,
            5.4,
            5.99,
            7.69,
            3.3,
            5.24
          ],
          [
            "PoseGraphNet (2026) [51] a",
            5.2,
            4.29,
            4.22,
            5.19,
            4.88,
            6.9,
            10.73,
            4.75,
            6.84,
            10.8,
            5.27,
            7.08,
            7.9,
            9.32,
            5.39,
            6.88,
            8.13,
            9.6,
            3.95,
            6.7
          ],
          [
            "Global - Local Network (2026) [30] a",
            6.65,
            6.38,
            7.26,
            8.08,
            7.49,
            10.06,
            17.18,
            7.37,
            10.1,
            17.0,
            6.71,
            8.08,
            9.27,
            10.72,
            6.7,
            8.06,
            9.37,
            10.91,
            6.95,
            9.18
          ],
          [
            "PAST-Net (ours)",
            2.99,
            2.42,
            2.71,
            2.7,
            2.99,
            4.26,
            6.11,
            2.84,
            4.26,
            6.07,
            3.05,
            4.27,
            4.95,
            6.47,
            3.06,
            4.3,
            4.97,
            6.6,
            2.49,
            4.08
          ]
        ]
      },
      "measured": {
        "columns": [
          "Method",
          "J0 (cm)",
          "J1 (cm)",
          "J2 (cm)",
          "J3 (cm)",
          "J4 (cm)",
          "J5 (cm)",
          "J6 (cm)",
          "J7 (cm)",
          "J8 (cm)",
          "J9 (cm)",
          "J10 (cm)",
          "J11 (cm)",
          "J12 (cm)",
          "J13 (cm)",
          "J14 (cm)",
          "J15 (cm)",
          "J16 (cm)",
          "J17 (cm)",
          "J18 (cm)",
          "Average (cm)"
        ],
        "rows": [
          [
            "DGCNN (2019) [62]",
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null
          ],
          [
            "PointTransformer (2021) [60]",
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null
          ],
          [
            "PointMLP (2022) [61]",
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null
          ],
          [
            "mmPose-FK (2024) [52] a,b",
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null
          ],
          [
            "DP-CNN–BiLSTM (2025) [21] a,c",
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null
          ],
          [
            "PntPoseAT (2025) [53] a",
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null
          ],
          [
            "PoseGraphNet (2026) [51] a",
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null
          ],
          [
            "Global - Local Network (2026) [30] a",
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null
          ],
          [
            "PAST-Net (ours)",
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null
          ]
        ]
      },
      "provenance": [],
      "reference_source": "paper/reference/table_05.csv",
      "reference_sha256": "b4b3ba9fe3b29ab88dde3a26ac637a990d89e15951b9a8116685c04542403560"
    },
    "6": {
      "id": "6",
      "title": "Five-seed robustness",
      "reference": {
        "columns": [
          "Method",
          "Seed 42",
          "Seed 50",
          "Seed 66",
          "Seed 69",
          "Seed 87",
          "Mean ± SD (cm)"
        ],
        "rows": [
          [
            "PntPoseAT",
            5.24,
            5.32,
            5.22,
            5.33,
            5.23,
            "5.27 ± 0.05"
          ],
          [
            "PAST-Net (ours)",
            4.09,
            4.09,
            4.06,
            4.09,
            4.05,
            "4.08 ± 0.02"
          ]
        ]
      },
      "measured": {
        "columns": [
          "Method",
          "Seed 42",
          "Seed 50",
          "Seed 66",
          "Seed 69",
          "Seed 87",
          "Mean ± SD (cm)"
        ],
        "rows": [
          [
            "PntPoseAT",
            null,
            null,
            null,
            null,
            null,
            null
          ],
          [
            "PAST-Net (ours)",
            null,
            null,
            null,
            null,
            null,
            null
          ]
        ]
      },
      "provenance": [],
      "reference_source": "paper/reference/table_06.csv",
      "reference_sha256": "2102f07be7edd10d8fa97d2f78a8ebd20e8dcf43e9ab13a273ec9db19bb52af8"
    },
    "7": {
      "id": "7",
      "title": "Split protocols",
      "reference": {
        "columns": [
          "Setting",
          "Method",
          "Sequence-chunk MLE (cm)",
          "Subject-independent MLE (cm)"
        ],
        "rows": [
          [
            "Multi-frame",
            "PAST-Net (ours)",
            4.08,
            4.53
          ],
          [
            "Multi-frame",
            "PntPoseAT",
            5.24,
            6.18
          ],
          [
            "Multi-frame",
            "mmPose-FK",
            5.57,
            6.3
          ],
          [
            "Multi-frame",
            "Dual-Path CNN–BiLSTM",
            8.19,
            9.38
          ],
          [
            "Multi-frame",
            "Global–Local Network",
            9.18,
            10.73
          ],
          [
            "Single-frame",
            "PointMLP",
            5.6,
            5.31
          ],
          [
            "Single-frame",
            "PoseGraphNet",
            6.7,
            5.99
          ],
          [
            "Single-frame",
            "PointTransformer",
            7.17,
            6.52
          ],
          [
            "Single-frame",
            "DGCNN",
            7.3,
            6.6
          ]
        ]
      },
      "measured": {
        "columns": [
          "Setting",
          "Method",
          "Sequence-chunk MLE (cm)",
          "Subject-independent MLE (cm)"
        ],
        "rows": [
          [
            "Multi-frame",
            "PAST-Net (ours)",
            null,
            null
          ],
          [
            "Multi-frame",
            "PntPoseAT",
            null,
            null
          ],
          [
            "Multi-frame",
            "mmPose-FK",
            null,
            null
          ],
          [
            "Multi-frame",
            "Dual-Path CNN–BiLSTM",
            null,
            null
          ],
          [
            "Multi-frame",
            "Global–Local Network",
            null,
            null
          ],
          [
            "Single-frame",
            "PointMLP",
            null,
            null
          ],
          [
            "Single-frame",
            "PoseGraphNet",
            null,
            null
          ],
          [
            "Single-frame",
            "PointTransformer",
            null,
            null
          ],
          [
            "Single-frame",
            "DGCNN",
            null,
            null
          ]
        ]
      },
      "provenance": [],
      "reference_source": "paper/reference/table_07.csv",
      "reference_sha256": "9441850dfee67ce14eb628d25f2138b004262b8832f8a7b55c7a89f2eeb0e243"
    },
    "8": {
      "id": "8",
      "title": "Module ablation",
      "reference": {
        "columns": [
          "Model configuration",
          "MLE (cm)"
        ],
        "rows": [
          [
            "Baseline",
            5.17
          ],
          [
            "Baseline + s patial module",
            4.39
          ],
          [
            "Baseline + s patial module + t emporal module",
            4.08
          ]
        ]
      },
      "measured": {
        "columns": [
          "Model configuration",
          "MLE (cm)"
        ],
        "rows": [
          [
            "Baseline",
            null
          ],
          [
            "Baseline + s patial module",
            null
          ],
          [
            "Baseline + s patial module + t emporal module",
            null
          ]
        ]
      },
      "provenance": [],
      "reference_source": "paper/reference/table_08.csv",
      "reference_sha256": "c6a10c6297a286a634f3677bcc005b8254deef9145b2946f012d89ecea77ae18"
    },
    "9": {
      "id": "9",
      "title": "Input features",
      "reference": {
        "columns": [
          "Input features",
          "MLE (cm)"
        ],
        "rows": [
          [
            "X , Y , Z",
            4.27
          ],
          [
            "X , Y , Z ,Doppler information , and SNR",
            4.08
          ]
        ]
      },
      "measured": {
        "columns": [
          "Input features",
          "MLE (cm)"
        ],
        "rows": [
          [
            "X , Y , Z",
            null
          ],
          [
            "X , Y , Z ,Doppler information , and SNR",
            null
          ]
        ]
      },
      "provenance": [],
      "reference_source": "paper/reference/table_09.csv",
      "reference_sha256": "2e0bdd0447dd81eaa0af30824f7bf24537c4d660c2ff56c489797f86c4eaf8c5"
    },
    "10": {
      "id": "10",
      "title": "Point count ablation",
      "reference": {
        "columns": [
          "Input points",
          "MLE (cm)"
        ],
        "rows": [
          [
            "25",
            5.23
          ],
          [
            "50",
            4.38
          ],
          [
            "75",
            4.21
          ],
          [
            "100",
            4.11
          ],
          [
            "125",
            4.11
          ],
          [
            "150",
            4.09
          ],
          [
            "300",
            4.08
          ]
        ]
      },
      "measured": {
        "columns": [
          "Input points",
          "MLE (cm)"
        ],
        "rows": [
          [
            "25",
            null
          ],
          [
            "50",
            null
          ],
          [
            "75",
            null
          ],
          [
            "100",
            null
          ],
          [
            "125",
            null
          ],
          [
            "150",
            null
          ],
          [
            "300",
            null
          ]
        ]
      },
      "provenance": [],
      "reference_source": "paper/reference/table_10.csv",
      "reference_sha256": "c0c67c9cdfcebdff577218540b3430a2c4e4fb5dc9536f04b1d0bc983e301319"
    }
  },
  "snapshot": {
    "schema_version": 1,
    "generated_at": "2026-10-27T06:40:47.227603+00:00",
    "selection": "Per-cell best compatible complete test: minimize MLE/PA-MPJPE, maximize PCK/AUC.",
    "missing_value": null,
    "records_scanned": 0,
    "records_included": [],
    "records_rejected": [],
    "identities": {}
  }
};
