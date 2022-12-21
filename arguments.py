from typing import Optional
from dataclasses import dataclass, field
from transformers import TrainingArguments

@dataclass
class ModelArguments:
    PLM: str = field(
        default="klue/roberta-large",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    model_name: str = field(
        default="RobertaForSequenceClassification",
        metadata={
            "help": "Model class name"
        },
    )
    

@dataclass
class DataTrainingArguments:
    max_length: int = field(
        default=128, 
        metadata={
            "help": "Max length of input sequence"
        },
    )
    data_dir: str = field(
        default="data", 
        metadata={
            "help": "path of data directory"
        }
    )
    train_data_file: str = field(
        default="train.csv", 
        metadata={
            "help": "name of train data"
        }
    )
    validation_ratio: Optional[float] = field(
        default=0.2, 
        metadata={
            "help": "validation ratio"
        }
    )
    minimum_size: Optional[int] = field(
        default=500, 
        metadata={
            "help": "minimum data size"
        }
    )
    maximum_size: Optional[int] = field(
        default=6000, 
        metadata={
            "help": "maximum data size"
        }
    )

@dataclass
class TrainingArguments(TrainingArguments):
    report_to: Optional[str] = field(default="wandb")
    output_dir: str = field(
        default="exps",
        metadata={
            "help": "model output directory"
        }
    )
    rdrop: bool = field(
        default=False,
        metadata={
            "help": "rdrop training"
        }
    )
    loss_fn: str = field(
        default="crossentropy",
        metadata={
            "help": "training loss"
        }
    )

@dataclass
class LoggingArguments:
    dotenv_path: Optional[str] = field(
        default="wandb.env", 
        metadata={
            "help": "input your dotenv path"
        },
    )
    project_name: Optional[str] = field(
        default="SKKU Sentence Classification", 
        metadata={
            "help": "project name"
        },
    )
    group_name: Optional[str] = field(
        default="Baseline", 
        metadata={
            "help": "group name"
        },
    )


@dataclass
class InferenceArguments:
    model_path: Optional[str] = field(
        default="./exps/checkpoint-500.pt", 
        metadata={
            "help": "trained model path"
        },
    )
    test_data_file: str = field(
        default="test.csv", 
        metadata={
            "help": "name of test data"
        }
    )
    submission_data_file: str = field(
        default="sample_submission.csv", 
        metadata={
            "help": "name of submission data"
        }
    )
    model_type: str = field(
        default="roberta", 
        metadata={
            "help": "model type"
        },
    )
    model_name: str = field(
        default="RobertaForSequenceClassification", 
        metadata={
            "help": "model class name"
        },
    )
    save_file_name: str = field(
        default="2022-12-16-01.csv", 
        metadata={
            "help": "name of saved submission file name"
        }
    )
    save_dir: str = field(
        default="results", 
        metadata={
            "help": "submission directory"
        },
    )