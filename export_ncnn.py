from transformers import AutoModelForSpeechSeq2Seq, WhisperFeatureExtractor
import torch
import torchaudio
import pnnx

# https://huggingface.co/openai/whisper-tiny/blob/main/config.json
model_name = "whisper-tiny"
num_mel_bins = 80
d_model = 384

# https://huggingface.co/openai/whisper-base/blob/main/config.json
# model_name = "whisper-base"
# num_mel_bins = 80
# d_model = 512

# https://huggingface.co/openai/whisper-small/blob/main/config.json
# model_name = "whisper-small"
# num_mel_bins = 80
# d_model = 768

# https://huggingface.co/openai/whisper-medium/blob/main/config.json
# model_name = "whisper-medium"
# num_mel_bins = 80
# d_model = 1024

# https://huggingface.co/openai/whisper-large-v3-turbo/blob/main/config.json
# model_name = "whisper-large-v3-turbo"
# num_mel_bins = 128
# d_model = 1280


model = AutoModelForSpeechSeq2Seq.from_pretrained('openai/' + model_name)
model.model.eval()

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/feature_extraction_whisper.py
# _torch_extract_fbank_features
class FbankDirectCallWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained('openai/' + model_name)

    def forward(self, waveform):
        window = torch.hann_window(self.feature_extractor.n_fft)

        power_spectrogram = torchaudio.functional.spectrogram(
            waveform,
            pad=0,
            window=window,
            n_fft=self.feature_extractor.n_fft,
            hop_length=self.feature_extractor.hop_length,
            win_length=self.feature_extractor.n_fft,
            power=2.0,
            normalized=False,
            center=True,
            pad_mode="reflect",
        )
        # power_spectrogram: [batch, 201, 3000] (real)

        # (n_mels, n_freq) @ (n_freq, n_time) -> (n_mels, n_time)
        mel_filters = torch.from_numpy(self.feature_extractor.mel_filters).to(torch.float32)

        mel_spec = mel_filters.T @ power_spectrogram

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()

        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec

fbank_model = FbankDirectCallWrapper()
fbank_model.eval()

dummy_waveform = torch.randn(1, 480000)
pnnx.export(fbank_model, model_name + "_fbank.pt", (dummy_waveform,))


# export encoder
def infer_forward_encoder(input_features):
    encoder_outputs = model.model.encoder(
        input_features=input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None)
    return encoder_outputs[0]

model.model.forward = infer_forward_encoder

dummy_input = torch.randn(1, num_mel_bins, 3000)
pnnx.export(model.model, model_name + "_encoder.pt", (dummy_input,))


# export embed token
def infer_forward_embed_token(ids):
    return model.model.decoder.embed_tokens(ids)

model.model.forward = infer_forward_embed_token

dummy_ids = torch.randint(0, 100, (1, 1))
pnnx.export(model.model, model_name + "_embed_token.pt", (dummy_ids,))


# export embed position
def infer_forward_embed_position(ids):
    return torch.nn.Embedding.forward(model.model.decoder.embed_positions, ids)

model.model.forward = infer_forward_embed_position

dummy_ids = torch.randint(0, 100, (1, 1))
pnnx.export(model.model, model_name + "_embed_position.pt", (dummy_ids,))


# export decoder
def infer_forward_decoder(inputs_embeds, encoder_hidden_states, attention_mask):
    decoder_outputs = model.model.decoder(
        input_ids=None,
        attention_mask=attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=inputs_embeds,
        position_ids=None,
        use_cache=False,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None)
    return decoder_outputs[0]

model.model.forward = infer_forward_decoder

dummy_embeds = torch.randn(1, 1, d_model)
dummy_encoder_out = torch.randn(1, 1500, d_model) # (B, Seq, Dim)
decoder_attention_mask = torch.randn(1, 1, 1, 1)
pnnx.export(model.model, model_name + "_decoder.pt", (dummy_embeds, dummy_encoder_out, decoder_attention_mask))


# export proj_out
dummy_hidden_state = torch.randn(1, 1, d_model)
pnnx.export(model.proj_out, model_name + "_proj_out.pt", (dummy_hidden_state,))



def modify_pnnx_file(filename):

    import re
    """
    Reads the provided pnnx.py file, applies specified modifications,
    and saves it as a new file.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        print(f"Please make sure '{filename}' is in the same directory as this script.")
        return

    new_lines = []
    for line in lines:
        new_line = line

        # 1. comment lines with "v_2.reshape(1, 1)"
        if "v_2.reshape(1, 1)" in new_line:
            new_line = "# " + new_line.lstrip() # lstrip() to handle indentation

        # 2. set attn_mask=v_2 for lines with "attn_mask="
        # This regex is safer as it won't affect other arguments on the same line.
        if "attn_mask=" in new_line:
            new_line = re.sub(r'attn_mask=[^,)]+', 'attn_mask=v_2', new_line)

        # 3. comment lines with "v_3 = self.pnnx_fold_"
        if "v_3 = self.pnnx_fold_" in new_line:
            new_line = "# " + new_line.lstrip()

        # 4. change "v_4 = (v_0 + v_3)" to v_4 = v_0
        if "v_4 = (v_0 + v_3)" in new_line:
            new_line = new_line.replace("(v_0 + v_3)", "v_0")

        # 5. change “v_2 = torch.rand(1, 1, 1, 1, dtype=torch.float)" to v_2 = torch.rand(1, 1, dtype=torch.float)
        # This modification needs to be applied in three places in the file.
        if "v_2 = torch.rand(1, 1, 1, 1, dtype=torch.float)" in new_line:
            new_line = new_line.replace(
                "torch.rand(1, 1, 1, 1, dtype=torch.float)",
                "torch.rand(1, 1, dtype=torch.float)"
            )

        new_lines.append(new_line)

    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"Successfully modified and overwritten the file '{filename}'.")

model_name_underscore = model_name.replace('-', '_')

modify_pnnx_file(model_name_underscore + "_decoder_pnnx.py")

import os
command_str = "python3 -c 'import " + model_name_underscore + "_decoder_pnnx; " + model_name_underscore + "_decoder_pnnx.export_torchscript()'"
os.system(command_str)
os.rename(model_name_underscore + "_decoder_pnnx.py.pt", model_name_underscore + "_decoder.pt")
command_str = "pnnx " + model_name_underscore + "_decoder.pt"
os.system(command_str)

def add_kv_cache_to_ncnn_param(filename):
    import os
    import re
    """
    Modifies an ncnn.param file to add a KV cache mechanism for all
    MultiHeadAttention layers and overwrites the original file.

    This version correctly handles the ncnn magic number at the start of the file.
    """
    if not os.path.exists(filename):
        print(f"Error: The file '{filename}' was not found.")
        return

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # --- Sanity check for file format ---
    if len(lines) < 2:
        print("Error: File is too short to be a valid ncnn.param file.")
        return

    try:
        # The first line is the magic number
        magic_number_line = lines[0]
        # The second line contains the counts
        header_line_index = 1
        header_parts = lines[header_line_index].strip().split()
        original_layer_count = int(header_parts[0])
        original_blob_count = int(header_parts[1])
    except (ValueError, IndexError):
        print("Error: Could not parse layer and blob counts from the file header.")
        return

    # --- Pass 1: Find all MultiHeadAttention layers ---
    mha_indices = [i for i, line in enumerate(lines) if line.strip().startswith("MultiHeadAttention")]
    mha_count = len(mha_indices)

    if mha_count == 0:
        print("No 'MultiHeadAttention' layers found. The file will not be modified.")
        return

    # --- Modify MultiHeadAttention layers ---
    for i, line_index in enumerate(mha_indices):
        parts = lines[line_index].strip().split()

        layer_type, layer_name, input_count_str, output_count_str = parts[:4]
        input_count, output_count = int(input_count_str), int(output_count_str)

        blob_and_params = parts[4:]
        inputs = blob_and_params[:input_count]
        outputs = blob_and_params[input_count : input_count + output_count]
        params = blob_and_params[input_count + output_count :]

        inputs.extend([f"cache_k{i}", f"cache_v{i}"])
        outputs.extend([f"out_cache_k{i}", f"out_cache_v{i}"])
        params.append("7=1")

        new_line_parts = [
            f"{layer_type:<24}", f"{layer_name:<24}",
            str(input_count + 2), str(output_count + 2),
            *inputs, *outputs, *params
        ]
        lines[line_index] = " ".join(new_line_parts) + "\n"

    # --- Modify header: +1 for the new Input layer, +4 blobs for each MHA ---
    new_layer_count = original_layer_count + 1
    new_blob_count = original_blob_count + (mha_count * 4)
    lines[header_line_index] = f"{new_layer_count} {new_blob_count}\n"

    # --- Add the new Input layer for KV cache ---
    # Find where to insert: after the initial block of Input/Split layers
    insert_pos = header_line_index + 1
    for i in range(insert_pos, len(lines)):
        if lines[i].strip().startswith("Input") or lines[i].strip().startswith("Split"):
            insert_pos = i + 1
        else:
            break

    cache_blob_names = [name for i in range(mha_count) for name in (f"cache_k{i}", f"cache_v{i}")]
    input_layer_line = (
        f"{'Input':<24} {'kv_cache':<24} 0 {len(cache_blob_names)} "
        f"{' '.join(cache_blob_names)}\n"
    )
    lines.insert(insert_pos, input_layer_line)

    # --- Write changes back to the file ---
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"Successfully added KV cache to {mha_count} MultiHeadAttention layers.")
    print(f"File '{filename}' has been modified and overwritten.")

add_kv_cache_to_ncnn_param(model_name_underscore + "_decoder.ncnn.param")

def update_gemm_params(param_file_path):
    import os
    import re
    """
    Reads an ncnn.param file, finds all lines starting with 'Gemm',
    and changes parameter '7=1' to '7=0' to support dynamic input.
    Overwrites the original file in-place.
    """
    if not os.path.isfile(param_file_path):
        print(f"Error: File '{param_file_path}' does not exist.")
        return

    # Read all lines from the param file
    with open(param_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    gemm_pattern = re.compile(r'^\s*Gemm\b')
    for line in lines:
        if gemm_pattern.match(line):
            # Only replace the exact '7=1' token, leave other '7=' occurrences untouched if any
            line = re.sub(r'(\b7=)1\b', r'\g<1>0', line)
        new_lines.append(line)

    # Write the modified lines back to the same file
    with open(param_file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"Updated '7=1' to '7=0' in all Gemm layers of '{param_file_path}'.")

update_gemm_params(model_name_underscore + "_decoder.ncnn.param")
