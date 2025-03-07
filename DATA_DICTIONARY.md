# Data Dictionary for MFCC features and Metadata

## Metadata
| Variable Name    | Description                        | Data Type | Example      |
| ---------------- | ---------------------------------- | --------- | ------------ |
| `cohort`         | Study group and participant cohort | String    | `EU150-001`  |
| `participant_id` | Unique participant identifier      | String    | `001`        |
| `date`           | Date of recording (YYYY-MM-DD)     | String    | `2023-11-14` |
| `time`           | Time of recording (HH:MM)          | String    | `17:56`      |

---

## MFCC Features (1-78)

### MFCC 1 (Features 1-6)
| feature_{n} | Human-Readable Name | Description                    | Data Type | Notes                               |
| ----------- | ------------------- | ------------------------------ | --------- | ----------------------------------- |
| `feature_1` | `mfcc1_mean`        | Mean of MFCC Coefficient 1     | Float     | Low-frequency spectral energy       |
| `feature_2` | `mfcc1_var`         | Variance of MFCC Coefficient 1 | Float     | Stability of low-frequency patterns |
| `feature_3` | `mfcc1_median`      | Median of MFCC Coefficient 1   | Float     |                                     |
| `feature_4` | `mfcc1_max`         | Maximum of MFCC Coefficient 1  | Float     |                                     |
| `feature_5` | `mfcc1_min`         | Minimum of MFCC Coefficient 1  | Float     |                                     |
| `feature_6` | `mfcc1_range`       | Range of MFCC Coefficient 1    | Float     | Max - Min                           |

### MFCC 2 (Features 7-12)
| feature_{n}  | Human-Readable Name | Description                    |
| ------------ | ------------------- | ------------------------------ |
| `feature_7`  | `mfcc2_mean`        | Mean of MFCC Coefficient 2     |
| `feature_8`  | `mfcc2_var`         | Variance of MFCC Coefficient 2 |
| `feature_9`  | `mfcc2_median`      | Median of MFCC Coefficient 2   |
| `feature_10` | `mfcc2_max`         | Maximum of MFCC Coefficient 2  |
| `feature_11` | `mfcc2_min`         | Minimum of MFCC Coefficient 2  |
| `feature_12` | `mfcc2_range`       | Range of MFCC Coefficient 2    |

### MFCC 3 (Features 13-18)
| feature_{n}  | Human-Readable Name | Description                    |
| ------------ | ------------------- | ------------------------------ |
| `feature_13` | `mfcc3_mean`        | Mean of MFCC Coefficient 3     |
| `feature_14` | `mfcc3_var`         | Variance of MFCC Coefficient 3 |
| `feature_15` | `mfcc3_median`      | Median of MFCC Coefficient 3   |
| `feature_16` | `mfcc3_max`         | Maximum of MFCC Coefficient 3  |
| `feature_17` | `mfcc3_min`         | Minimum of MFCC Coefficient 3  |
| `feature_18` | `mfcc3_range`       | Range of MFCC Coefficient 3    |

### MFCC 4 (Features 19-24)
| feature_{n}  | Human-Readable Name | Description                    |
| ------------ | ------------------- | ------------------------------ |
| `feature_19` | `mfcc4_mean`        | Mean of MFCC Coefficient 4     |
| `feature_20` | `mfcc4_var`         | Variance of MFCC Coefficient 4 |
| `feature_21` | `mfcc4_median`      | Median of MFCC Coefficient 4   |
| `feature_22` | `mfcc4_max`         | Maximum of MFCC Coefficient 4  |
| `feature_23` | `mfcc4_min`         | Minimum of MFCC Coefficient 4  |
| `feature_24` | `mfcc4_range`       | Range of MFCC Coefficient 4    |

### MFCC 5 (Features 25-30)
| feature_{n}  | Human-Readable Name | Description                    |
| ------------ | ------------------- | ------------------------------ |
| `feature_25` | `mfcc5_mean`        | Mean of MFCC Coefficient 5     |
| `feature_26` | `mfcc5_var`         | Variance of MFCC Coefficient 5 |
| `feature_27` | `mfcc5_median`      | Median of MFCC Coefficient 5   |
| `feature_28` | `mfcc5_max`         | Maximum of MFCC Coefficient 5  |
| `feature_29` | `mfcc5_min`         | Minimum of MFCC Coefficient 5  |
| `feature_30` | `mfcc5_range`       | Range of MFCC Coefficient 5    |

### MFCC 6 (Features 31-36)
| feature_{n}  | Human-Readable Name | Description                    |
| ------------ | ------------------- | ------------------------------ |
| `feature_31` | `mfcc6_mean`        | Mean of MFCC Coefficient 6     |
| `feature_32` | `mfcc6_var`         | Variance of MFCC Coefficient 6 |
| `feature_33` | `mfcc6_median`      | Median of MFCC Coefficient 6   |
| `feature_34` | `mfcc6_max`         | Maximum of MFCC Coefficient 6  |
| `feature_35` | `mfcc6_min`         | Minimum of MFCC Coefficient 6  |
| `feature_36` | `mfcc6_range`       | Range of MFCC Coefficient 6    |

### MFCC 7 (Features 37-42)
| feature_{n}  | Human-Readable Name | Description                    |
| ------------ | ------------------- | ------------------------------ |
| `feature_37` | `mfcc7_mean`        | Mean of MFCC Coefficient 7     |
| `feature_38` | `mfcc7_var`         | Variance of MFCC Coefficient 7 |
| `feature_39` | `mfcc7_median`      | Median of MFCC Coefficient 7   |
| `feature_40` | `mfcc7_max`         | Maximum of MFCC Coefficient 7  |
| `feature_41` | `mfcc7_min`         | Minimum of MFCC Coefficient 7  |
| `feature_42` | `mfcc7_range`       | Range of MFCC Coefficient 7    |

### MFCC 8 (Features 43-48)
| feature_{n}  | Human-Readable Name | Description                    |
| ------------ | ------------------- | ------------------------------ |
| `feature_43` | `mfcc8_mean`        | Mean of MFCC Coefficient 8     |
| `feature_44` | `mfcc8_var`         | Variance of MFCC Coefficient 8 |
| `feature_45` | `mfcc8_median`      | Median of MFCC Coefficient 8   |
| `feature_46` | `mfcc8_max`         | Maximum of MFCC Coefficient 8  |
| `feature_47` | `mfcc8_min`         | Minimum of MFCC Coefficient 8  |
| `feature_48` | `mfcc8_range`       | Range of MFCC Coefficient 8    |

### MFCC 9 (Features 49-54)
| feature_{n}  | Human-Readable Name | Description                    |
| ------------ | ------------------- | ------------------------------ |
| `feature_49` | `mfcc9_mean`        | Mean of MFCC Coefficient 9     |
| `feature_50` | `mfcc9_var`         | Variance of MFCC Coefficient 9 |
| `feature_51` | `mfcc9_median`      | Median of MFCC Coefficient 9   |
| `feature_52` | `mfcc9_max`         | Maximum of MFCC Coefficient 9  |
| `feature_53` | `mfcc9_min`         | Minimum of MFCC Coefficient 9  |
| `feature_54` | `mfcc9_range`       | Range of MFCC Coefficient 9    |

### MFCC 10 (Features 55-60)
| feature_{n}  | Human-Readable Name | Description                     |
| ------------ | ------------------- | ------------------------------- |
| `feature_55` | `mfcc10_mean`       | Mean of MFCC Coefficient 10     |
| `feature_56` | `mfcc10_var`        | Variance of MFCC Coefficient 10 |
| `feature_57` | `mfcc10_median`     | Median of MFCC Coefficient 10   |
| `feature_58` | `mfcc10_max`        | Maximum of MFCC Coefficient 10  |
| `feature_59` | `mfcc10_min`        | Minimum of MFCC Coefficient 10  |
| `feature_60` | `mfcc10_range`      | Range of MFCC Coefficient 10    |

### MFCC 11 (Features 61-66)
| feature_{n}  | Human-Readable Name | Description                     |
| ------------ | ------------------- | ------------------------------- |
| `feature_61` | `mfcc11_mean`       | Mean of MFCC Coefficient 11     |
| `feature_62` | `mfcc11_var`        | Variance of MFCC Coefficient 11 |
| `feature_63` | `mfcc11_median`     | Median of MFCC Coefficient 11   |
| `feature_64` | `mfcc11_max`        | Maximum of MFCC Coefficient 11  |
| `feature_65` | `mfcc11_min`        | Minimum of MFCC Coefficient 11  |
| `feature_66` | `mfcc11_range`      | Range of MFCC Coefficient 11    |

### MFCC 12 (Features 67-72)
| feature_{n}  | Human-Readable Name | Description                     |
| ------------ | ------------------- | ------------------------------- |
| `feature_67` | `mfcc12_mean`       | Mean of MFCC Coefficient 12     |
| `feature_68` | `mfcc12_var`        | Variance of MFCC Coefficient 12 |
| `feature_69` | `mfcc12_median`     | Median of MFCC Coefficient 12   |
| `feature_70` | `mfcc12_max`        | Maximum of MFCC Coefficient 12  |
| `feature_71` | `mfcc12_min`        | Minimum of MFCC Coefficient 12  |
| `feature_72` | `mfcc12_range`      | Range of MFCC Coefficient 12    |

### MFCC 13 (Features 73-78)
| feature_{n}  | Human-Readable Name | Description                     |
| ------------ | ------------------- | ------------------------------- |
| `feature_73` | `mfcc13_mean`       | Mean of MFCC Coefficient 13     |
| `feature_74` | `mfcc13_var`        | Variance of MFCC Coefficient 13 |
| `feature_75` | `mfcc13_median`     | Median of MFCC Coefficient 13   |
| `feature_76` | `mfcc13_max`        | Maximum of MFCC Coefficient 13  |
| `feature_77` | `mfcc13_min`        | Minimum of MFCC Coefficient 13  |
| `feature_78` | `mfcc13_range`      | Range of MFCC Coefficient 13    |

---

## Notes
1. **MFCCs**:  
   - Coefficients 1-13 represent spectral characteristics of voice.  
   - Lower coefficients (1-6) capture broader spectral trends (low frequencies).  
   - Higher coefficients (7-13) capture finer details (high frequencies).  

2. **Statistics**:  
   - **Mean**: Average value over time.  
   - **Variance**: Stability of values (higher = more variability).  
   - **Range**: Difference between maximum and minimum values.  
