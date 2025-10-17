# ncnn-android-whisper

whisper语音识别ASR大模型的ncnn粗糙的实现流程 https://zhuanlan.zhihu.com/p/1962562831795857091


## guidelines for converting whisper models

### convert whisper checkpoints to ncnn models

```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip3 install transformers
pip3 install pnnx

python export_ncnn.py
```
