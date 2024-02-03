import logging
import os
from fastchat.conversation import (
    SeparatorStyle,
    get_conv_template,
)
from logging.handlers import RotatingFileHandler


chatgpt_conv = get_conv_template("chatgpt")
chatgpt_conv.set_system_message(
    "You are a helpful, respectful and honest assistant."
)
chatgpt_conv.sep_style = SeparatorStyle.ADD_COLON_SINGLE
chatgpt_conv.sep = "\n"

all_convs = {
    "chatglm2": get_conv_template("chatglm2"),
    "chatglm3": get_conv_template("chatglm3"),
    "llama2": get_conv_template("llama-2"),
    "vicuna": get_conv_template("vicuna_v1.1"),
    "zephyr": get_conv_template("zephyr"),
    "openchat": get_conv_template("openchat_3.5"),
    "openhermes": get_conv_template("OpenHermes-2.5-Mistral-7B"),
    "qwen": get_conv_template("qwen-7b-chat"),
    "mistral": get_conv_template("mistral"),
    "chatgpt": chatgpt_conv,
}

level_map = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}
config = {
    "llama2-7b": {
        "path": "meta-llama/Llama-2-7b-hf",
        "max_context_len": 4096,
        "chat_template": all_convs["llama2"],
        "use_flash_attn": True,
        "end_tokens": [
            "</s>",
        ],
    },
    "llama2-13b": {
        "path": "meta-llama/Llama-2-13b-hf",
        "max_context_len": 4096,
        "chat_template": all_convs["llama2"],
        "use_flash_attn": True,
        "end_tokens": [
            "</s>",
        ],
    },
    "llama2-70b": {
        "path": "meta-llama/Llama-2-70b-hf",
        "max_context_len": 4096,
        "chat_template": all_convs["llama2"],
        "use_flash_attn": True,
        "end_tokens": [
            "</s>",
        ],
    },
    "llama2-chat-7b": {
        "path": "meta-llama/Llama-2-7b-chat-hf",
        "max_context_len": 4096,
        "chat_template": all_convs["llama2"],
        "use_flash_attn": True,
        "end_tokens": [
            "</s>",
        ],
    },
    "llama2-chat-13b": {
        "path": "meta-llama/Llama-2-13b-chat-hf",
        "max_context_len": 4096,
        "chat_template": all_convs["llama2"],
        "use_flash_attn": True,
        "end_tokens": [
            "</s>",
        ],
    },
    "llama2-chat-70b": {
        "path": "meta-llama/Llama-2-70b-chat-hf",
        "max_context_len": 4096,
        "chat_template": all_convs["llama2"],
        "use_flash_attn": True,
        "end_tokens": [
            "</s>",
        ],
    },
    "vicuna-7b": {
        "path": "lmsys/vicuna-7b-v1.5",
        "max_context_len": 4096,
        "chat_template": all_convs["vicuna"],
        "use_flash_attn": True,
        "end_tokens": ["</s>", "USER:"],
    },
    "vicuna-13b": {
        "path": "lmsys/vicuna-13b-v1.5",
        "max_context_len": 4096,
        "chat_template": all_convs["vicuna"],
        "use_flash_attn": True,
        "end_tokens": ["</s>", "USER:"],
    },
    "vicuna-7b-16k": {
        "path": "lmsys/vicuna-7b-v1.5-16k",
        "max_context_len": 16000,
        "chat_template": all_convs["vicuna"],
        "use_flash_attn": True,
        "end_tokens": ["</s>", "USER:"],
    },
    "vicuna-13b-16k": {
        "path": "lmsys/vicuna-13b-v1.5-16k",
        "max_context_len": 16000,
        "chat_template": all_convs["vicuna"],
        "use_flash_attn": True,
        "end_tokens": ["</s>", "USER:"],
    },
    "chatglm2-6b-32k": {
        "path": "THUDM/chatglm2-6b-32k",
        "max_context_len": 32000,
        "chat_template": all_convs["chatglm2"],
        "use_flash_attn": False,
        "end_tokens": ["问："],
    },
    "chatglm3-6b-32k": {
        "path": "THUDM/chatglm3-6b-32k",
        "max_context_len": 32000,
        "chat_template": all_convs["chatglm3"],
        "use_flash_attn": False,
        "end_tokens": ["<|user|>"],
    },
    "chatglm3-6b": {
        "path": "THUDM/chatglm3-6b",
        "max_context_len": 8192,
        "chat_template": all_convs["chatglm3"],
        "use_flash_attn": False,
        "end_tokens": ["<|user|>"],
    },
    "qwen-chat-7b": {
        "path": "Qwen/Qwen-7B-Chat",
        "max_context_len": 8192,
        "chat_template": all_convs["qwen"],
        "use_flash_attn": False,
        "end_tokens": ["<|im_end|>"],
    },
    "qwen-chat-14b": {
        "path": "Qwen/Qwen-14B-Chat",
        "max_context_len": 8192,
        "chat_template": all_convs["qwen"],
        "use_flash_attn": False,
        "end_tokens": ["<|im_end|>"],
    },
    "qwen-chat-72b": {
        "path": "Qwen/Qwen-72B-Chat",
        "max_context_len": 8192,
        "chat_template": all_convs["qwen"],
        "use_flash_attn": False,
        "end_tokens": ["<|im_end|>"],
    },
    "zephyr-7b-beta": {
        "path": "HuggingFaceH4/zephyr-7b-beta",
        "max_context_len": 8192,
        "chat_template": all_convs["zephyr"],
        "use_flash_attn": True,
        "end_tokens": ["</s>", "<|user|>"],
    },
    "openchat-3.5": {
        "path": "openchat/openchat_3.5",
        "max_context_len": 8192,
        "chat_template": all_convs["openchat"],
        "use_flash_attn": True,
        "end_tokens": ["<|end_of_turn|>", "GPT4 Correct User"],
    },
    "openhermes-2.5": {
        "path": "teknium/OpenHermes-2.5-Mistral-7B",
        "max_context_len": 8192,
        "chat_template": all_convs["openhermes"],
        "use_flash_attn": True,
        "end_tokens": ["<|im_end|>", "user\n"],
    },
    "starling-7b": {
        "path": "berkeley-nest/Starling-LM-7B-alpha",
        "max_context_len": 8192,
        "chat_template": all_convs["openhermes"],
        "use_flash_attn": True,
        "end_tokens": ["<|end_of_turn|>", "GPT4 Correct User"],
    },
    "gpt-3.5-turbo": {
        "path": "",
        "max_context_len": 4096,
        "chat_template": all_convs["chatgpt"],
        "end_tokens": [],
    },
    "gpt-3.5-turbo-16k": {
        "path": "",
        "max_context_len": 16000,
        "chat_template": all_convs["chatgpt"],
        "end_tokens": [],
    },
    "gpt-4": {
        "path": "",
        "max_context_len": 8192,
        "chat_template": all_convs["chatgpt"],
        "end_tokens": [],
    },
    "mixtral-instruct-v0.1": {
        "path": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "max_context_len": 32768,
        "chat_template": all_convs["mistral"],
        "use_flash_attn": True,
        "end_tokens": ["</s>", "[INST]"],
    },
    "mistral-instruct-v0.2": {
        "path": "mistralai/Mistral-7B-Instruct-v0.2",
        "max_context_len": 32768,
        "chat_template": all_convs["mistral"],
        "use_flash_attn": True,
        "end_tokens": ["</s>", "[INST]"],
    },
}


def get_logger(
    name: str,
    logger_level: str = None,
    console_level: str = None,
    file_level: str = None,
    log_path: str = None,
    maxBytes: int = 1e8,
    backupCount: int = 1,
):
    """Configure the logger and return it.

    Args:
        name (str, optional): Name of the logger, usually __name__.
            Defaults to None. None refers to root logger, usually useful
            when setting the default config at the top level script.
        logger_level (str, optional): level of logger. Defaults to None.
            None is treated as `debug`.
        console_level (str, optional): level of console. Defaults to None.
        file_level (str, optional): level of file. Defaults to None.
            None is treated `debug`.
        log_path (str, optional): The path of the log.
        maxBytes (int): The maximum size of the log file.
            Only used if log_path is not None.
        backupCount (int): Number of rolling backup log files.
            If log_path is `app.log` and backupCount is 3, we will have
            `app.log`, `app.log.1`, `app.log.2` and `app.log.3`.
            Only used if log_path is not None.

    Note that console_level should only be used when configuring the
    root logger.
    """

    logger = logging.getLogger(name)
    if name:
        logger.setLevel(level_map[logger_level or "debug"])
    else:  # root logger default lv should be high to avoid external lib log
        logger.setLevel(level_map[logger_level or "warning"])

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # set up the logfile handler
    if log_path:
        root_logger = logger.root
        # logTime = datetime.datetime.now()
        # fn1, fn2 = os.path.splitext(log_path)
        # log_filename = f"{fn1}-{logTime.strftime('%Y%m%d-%H%M')}{fn2}"
        log_filename = log_path
        if os.path.dirname(log_filename):
            os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        # fh = logging.FileHandler(log_filename)
        fh = [
            handler
            for handler in root_logger.handlers
            if type(handler) == logging.FileHandler
        ]
        if fh:
            # global_logger.info(
            #     "Replaced the original root filehandler with new one."
            # )
            fh = fh[0]
            root_logger.removeHandler(fh)
        fh = RotatingFileHandler(
            filename=log_filename, maxBytes=maxBytes, backupCount=backupCount
        )
        fh.setLevel(level_map[file_level or "debug"])
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)

    # set up the console/stream handler
    # if name and console_level:
    #     raise ValueError(
    #         "`console_level` should only be set when configuring root logger."
    #     )
    if console_level:
        root_logger = logger.root
        sh = [
            handler
            for handler in root_logger.handlers
            if type(handler) == logging.StreamHandler
        ]
        if sh:
            # global_logger.info(
            #     "Replaced the original root streamhandler with new one."
            # )
            sh = sh[0]
            root_logger.removeHandler(sh)
        sh = logging.StreamHandler()
        sh.setLevel(level_map[console_level or "debug"])
        sh.setFormatter(formatter)
        root_logger.addHandler(sh)
    return logger
