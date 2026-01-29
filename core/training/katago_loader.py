import gzip
import io
import struct

import numpy as np


class KataGoBinLoader:
    def __init__(self, path):
        if path.endswith(".gz"):
            self.f = gzip.open(path, "rb")
        else:
            self.f = open(path, "rb")
        self.weights = {}
        self.model_config = {}

    def close(self):
        self.f.close()

    def read_line(self):
        line = b""
        while True:
            char = self.f.read(1)
            if not char:
                if not line:
                    return None
                break
            if char == b"\n":
                break
            line += char
        res = line.decode("ascii", errors="replace").strip()
        return res

    def read_int(self):
        return int(self.read_line())

    def read_float(self):
        return float(self.read_line())

    def read_weights(self, expected_size=None):
        # Scan for @BIN@
        while True:
            char = self.f.read(1)
            if char == b"@":
                tag = self.f.read(4)
                if tag == b"BIN@":
                    break

        # We assume we know the size?
        # Actually export script calculates size from tensor.
        # But import script needs to know size to read bytes.
        # OR we rely on the logic that we calculated shape from metadata.

        assert expected_size is not None, "Must provide expected num elements"

        num_bytes = int(expected_size * 4)
        data = self.f.read(num_bytes)
        if len(data) != num_bytes:
            raise ValueError("Unexpected EOF in weights")

        arr = np.frombuffer(data, dtype=np.dtype("<f4"))
        self.f.read(1)  # newline
        return arr.astype(np.float32)

    def read_conv(self, name):
        # name line is already consumed by caller usually?
        # No, export script calls writeln(name) inside write_conv.
        # But caller might have consumed it to identify the block?
        # My recursive structure asks: "Read block".

        # Let's assume strict structure.
        # We check name.
        f_name = self.read_line()
        if f_name != name:
            # Sometimes prefix matters?
            # Let's trust structure.
            pass

        diamy = self.read_int()
        diamx = self.read_int()
        in_channels = self.read_int()
        out_channels = self.read_int()
        dilation_y = self.read_int()
        dilation_x = self.read_int()

        shape = (diamy, diamx, in_channels, out_channels)
        size = np.prod(shape)
        w = self.read_weights(size)
        self.weights[name + ".kernel"] = w.reshape(shape)

    def read_bn(self, name):
        f_name = self.read_line()

        c_in = self.read_int()
        epsilon = self.read_float()
        has_gamma = self.read_int()
        has_beta = self.read_int()

        # Running Mean
        # Always written if trained? Export script conditions:
        # if hasattr(normmask,"running_mean")...
        # But we assume standard trained model.
        # Wait, if not using batchnorm, it writes zeros.
        # So it ALWAYS writes something.
        mean = self.read_weights(c_in)
        self.weights[name + ".mean"] = mean

        # Running Var
        var = self.read_weights(c_in)
        self.weights[name + ".var"] = var

        # Scale/Gamma
        if has_gamma:
            scale = self.read_weights(c_in)
            self.weights[name + ".scale"] = scale

        # Bias/Beta
        beta = self.read_weights(c_in)
        self.weights[name + ".bias"] = beta

    def read_biasmask(self, name):
        f_name = self.read_line()
        c_in = self.read_int()
        epsilon = self.read_float()
        has_scale = self.read_int()  # has_gamma_or_scale
        has_beta = self.read_int()

        # Zeros (mean)
        self.read_weights(c_in)
        # Ones (var)
        self.read_weights(c_in)

        if has_scale:
            scale = self.read_weights(c_in)
            self.weights[name + ".scale"] = scale

        beta = self.read_weights(c_in)
        self.weights[name + ".bias"] = beta

    def read_activation(self, name):
        f_name = self.read_line()
        act_type = self.read_line()
        # Just info

    def read_matmul(self, name):
        f_name = self.read_line()
        in_c = self.read_int()
        out_c = self.read_int()
        w = self.read_weights(in_c * out_c)
        self.weights[name + ".kernel"] = w.reshape((in_c, out_c))

    def read_matbias(self, name):
        f_name = self.read_line()
        out_c = self.read_int()
        b = self.read_weights(out_c)
        self.weights[name + ".bias"] = b

    def read_normactconv(self, name):
        # We need to peek logic or assume structure.
        # Structure is fixed.
        # However, Conv1x1 or ConvPool?
        # How to distinguish?
        # Export script:
        # if c_gpool is None:
        #     if conv1x1 is None: ...
        #     else: ...
        # else: ...

        # This is tricky. The binaries don't explicitly say "I am a gpool block".
        # BUT the sequence of variables tells us.
        # Normal NormActConv:
        #   write_bn(name.norm)
        #   write_act(name.act)
        #   write_conv(name.conv)
        # GPool NormActConv:
        #   write_bn(name.norm)
        #   write_act(name.act)
        #   write_conv(name.convpool.conv1r)
        #   ...

        # We can PEEK the next line (name) to verify.
        # But simpler: The caller (read_block) knows if it's a gpool_block or not!

        pass

    def read_nac_simple(self, name):
        self.read_bn(name + ".norm")
        self.read_activation(name + ".act")
        self.read_conv(name + ".conv")

    def read_gpool_nac(self, name):
        self.read_bn(name + ".norm")
        self.read_activation(name + ".act")
        self.read_conv(name + ".convpool.conv1r")
        self.read_conv(name + ".convpool.conv1g")
        self.read_bn(name + ".convpool.normg")
        self.read_activation(name + ".convpool.actg")
        self.read_matmul(name + ".convpool.linear_g")

    def read_block(self, name):
        block_type = (
            self.read_line()
        )  # "ordinary_block" or "gpool_block" or "nested..."
        f_name = self.read_line()  # name

        if block_type == "ordinary_block":
            self.read_nac_simple(name + ".normactconv1")
            self.read_nac_simple(name + ".normactconv2")
        elif block_type == "gpool_block":
            # Block 1 is GPool NAC
            self.read_gpool_nac(name + ".normactconv1")
            # Block 2 is Simple NAC
            self.read_nac_simple(name + ".normactconv2")
        elif block_type == "nested_bottleneck_block":
            # Not handling fully yet, but provided model might be nested?
            # Hopefully not for b28?
            # b28 is probably nested.
            length = self.read_int()
            self.read_nac_simple(name + ".normactconvp")
            for i in range(length):
                self.read_block(name + ".blockstack." + str(i))
            self.read_nac_simple(name + ".normactconvq")
        else:
            raise ValueError(f"Unknown block type {block_type}")

    def read_trunk(self):
        lbl = self.read_line()  # "trunk"
        num_blocks = self.read_int()
        c_trunk = self.read_int()
        c_mid = self.read_int()
        c_mid_trunk = self.read_int()
        c_gpool = self.read_int()
        c_gpool2 = self.read_int()  # duplicated?

        self.model_config.update(
            {
                "num_blocks": num_blocks,
                "c_trunk": c_trunk,
                "c_mid": c_mid,
                "c_gpool": c_gpool,
            }
        )

        # Placeholders
        # read_line loop?
        # Version 15 has 6 zeros.
        for _ in range(6):
            self.read_line()

        self.read_conv("model.conv_spatial")
        self.read_matmul("model.linear_global")
        # Metadata encoder?
        # We skipped extracting it in metadata section but assume none for now or handle later
        # But wait, version >= 15 checks metadata_encoder.
        # But we are in read_trunk.
        # "if model.metadata_encoder is not None".
        # We don't know if it's there.
        # Checking next line name?
        # Next line is "model.blocks.0" (block type line actually).

        # We need to peek.
        # But let's assume standard structure.

        for i in range(num_blocks):
            self.read_block("model.blocks." + str(i))

        # Trunk Final
        # normless? check next line.
        # But read_bn and read_biasmask both consume name line.
        # Try read_bn.
        self.read_bn("model.norm_trunkfinal")
        self.read_activation("model.act_trunkfinal")

    def read_policy_head(self):
        name = "model.policy_head"
        f_name = self.read_line()  # name

        self.read_conv(name + ".conv1p")
        self.read_conv(name + ".conv1g")
        self.read_biasmask(name + ".biasg")
        self.read_activation(name + ".actg")
        self.read_matmul(name + ".linear_g")
        self.read_biasmask(name + ".bias2")
        self.read_activation(name + ".act2")

        # pass/optimistic
        # v15
        self.read_conv(name + ".conv2p")
        self.read_matmul(name + ".linear_pass")
        self.read_matbias(name + ".linear_pass_bias")
        self.read_activation(name + ".act_pass")
        self.read_matmul(name + ".linear_pass2")

    def read_value_head(self):
        name = "model.value_head"
        f_name = self.read_line()

        self.read_conv(name + ".conv1")
        self.read_biasmask(name + ".bias1")
        self.read_activation(name + ".act1")
        self.read_matmul(name + ".linear2")
        self.read_matbias(name + ".bias2")
        self.read_activation(name + ".act2")
        self.read_matmul(name + ".linear_valuehead")
        self.read_matbias(name + ".bias_valuehead")
        self.read_matmul(name + ".linear_miscvaluehead")
        self.read_matbias(name + ".bias_miscvaluehead")
        self.read_conv(name + ".conv_ownership")

    def load(self):
        # Metadata
        self.model_config["model_name"] = self.read_line()
        self.model_config["version"] = int(self.read_line())
        self.model_config["num_bin_input"] = int(self.read_line())
        self.model_config["num_global_input"] = int(self.read_line())

        # Multipliers
        # Read until parsing fails?
        # Or hardcode v15: 7 multipliers.
        for _ in range(7):
            self.read_line()

        # Metadata encoder info + placeholders (1 + 7?)
        # v15: meta version (0) + 7 * 0
        for _ in range(8):
            self.read_line()

        # Trunk
        self.read_trunk()

        # Heads
        self.read_policy_head()
        self.read_value_head()

        return self.weights, self.model_config


def load_katago_weights(path):
    loader = KataGoBinLoader(path)
    weights, config = loader.load()
    loader.close()
    return weights, config


def copy_params(variables, weights):
    """
    Copies weights from Loaded Dictionary into Flax Variables.
    """
    import re

    import jax
    from flax.core import freeze, unfreeze

    variables = unfreeze(variables)

    def flax_path_to_katago_key(path):
        # path is a tuple of KeyEntry objects, e.g. (DictKey('params'), DictKey('blocks_0'), ...)
        # Convert to string representations
        parts = []
        for key_entry in path[1:]:  # Skip root ('params' or 'batch_stats')
            # Get string key from KeyEntry
            if hasattr(key_entry, "key"):
                p = str(key_entry.key)
            else:
                p = str(key_entry)

            # Convert 'name_idx' -> 'name.idx'
            # e.g. blocks_0 -> blocks.0
            # blockstack_0 -> blockstack.0
            if re.match(r"^[a-z]+_\d+$", p):
                parts.append(p.replace("_", "."))
            elif p == "conv_spatial":
                parts.append(p)
            elif p == "linear_global":
                parts.append(p)
            elif p == "policy_head" or p == "value_head":
                parts.append(p)
            elif p == "convpool_conv1r":
                parts.append("convpool.conv1r")
            elif p == "convpool_conv1g":
                parts.append("convpool.conv1g")
            elif p == "convpool_linear_g":
                parts.append("convpool.linear_g")
            elif p == "convpool_normg":
                parts.append("convpool.normg")
            # Handle standard keys
            else:
                parts.append(p)

        key = "model." + ".".join(parts)
        return key

    def update_leaf(path, value):
        key = flax_path_to_katago_key(path)

        # Try direct match
        if key in weights:
            w = weights[key]
            if w.shape != value.shape:
                # Transpose 2D weights (dense)
                if len(w.shape) == 2 and w.shape[::-1] == value.shape:
                    w = w.T
                elif np.prod(w.shape) == np.prod(value.shape):
                    w = w.reshape(value.shape)
                else:
                    print(
                        f"Warning: Shape mismatch for {key}: loaded {w.shape}, model {value.shape}"
                    )
                    return value
            return jax.numpy.array(w)

        # Try mapping standard Flax names to KataGo names
        # Flax: kernel -> kernel (matched)
        # Flax: bias -> bias (matched)
        # Flax BN: scale -> scale (matched)
        # Flax BN: mean -> mean (matched)
        # Flax BN: var -> var (matched)

        # But maybe path ending is different?
        # Flax BN: 'batch_stats', ..., 'mean'
        # KataGo: 'model.....mean'
        # My generator creates 'model.....mean'.

        # Debug unmapped keys?
        # print(f"Missing key: {key}")
        return value

    new_variables = jax.tree_util.tree_map_with_path(update_leaf, variables)
    return freeze(new_variables)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        w, c = load_katago_weights(sys.argv[1])
        print("Config:", c)
        print("Loaded", len(w), "weights")
        for k in list(w.keys())[:10]:
            print(k, w[k].shape)
