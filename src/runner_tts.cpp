/*
 * nrvna ai - TTS Runner (OuteTTS + WavTokenizer)
 * Copyright (c) 2025 Sanmathi Bharamgouda
 * SPDX-License-Identifier: MIT
 *
 * Spectral ops (irfft, fold, embd_to_audio), text preprocessing, WAV format,
 * and default speaker profile adapted from llama.cpp tools/tts/tts.cpp.
 */

#define _USE_MATH_DEFINES

#include "nrvna/runner_tts.hpp"
#include "nrvna/logger.hpp"
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <regex>
#include <thread>

namespace nrvnaai {

// Static member definitions — TTS model is separate from text/vision model
std::shared_ptr<llama_model> TtsRunner::shared_tts_model_ = nullptr;
std::shared_ptr<llama_model> TtsRunner::shared_vocoder_ = nullptr;
std::string TtsRunner::current_tts_model_path_ = "";
std::string TtsRunner::current_vocoder_path_ = "";
std::mutex TtsRunner::tts_model_mutex_;
TtsVersion TtsRunner::detected_version_ = TtsVersion::V0_2;
std::string TtsRunner::v3_audio_text_;
std::string TtsRunner::v3_audio_data_;

namespace {

// ============================================================================
// Env helpers
// ============================================================================

static int env_int(const char* name, int defv) {
    if (const char* v = std::getenv(name)) return std::atoi(v);
    return defv;
}

// ============================================================================
// Spectral ops (from tts.cpp — pure math, safe to borrow)
// ============================================================================

void fill_hann_window(int length, bool periodic, float* output) {
    int offset = periodic ? 0 : -1;
    for (int i = 0; i < length; i++) {
        output[i] = 0.5f * (1.0f - cosf((2.0f * static_cast<float>(M_PI) * i) / (length + offset)));
    }
}

void twiddle(float* real, float* imag, int k, int N) {
    float angle = 2.0f * static_cast<float>(M_PI) * k / N;
    *real = cosf(angle);
    *imag = sinf(angle);
}

void irfft(int n, const float* inp_cplx, float* out_real) {
    int N = n / 2 + 1;
    std::vector<float> real_input(N), imag_input(N);
    for (int i = 0; i < N; ++i) {
        real_input[i] = inp_cplx[2 * i];
        imag_input[i] = inp_cplx[2 * i + 1];
    }

    std::vector<float> real_output(n), imag_output(n);
    for (int k = 0; k < n; ++k) {
        real_output[k] = 0.0f;
        imag_output[k] = 0.0f;
        for (int m = 0; m < N; ++m) {
            float tw_r, tw_i;
            twiddle(&tw_r, &tw_i, k * m, n);
            real_output[k] += real_input[m] * tw_r - imag_input[m] * tw_i;
            imag_output[k] += real_input[m] * tw_i + imag_input[m] * tw_r;
        }
    }
    for (int i = 0; i < n; ++i) {
        out_real[i] = real_output[i] / N;
    }
}

void fold(const std::vector<float>& data, int64_t n_out, int64_t n_win,
          int64_t n_hop, int64_t n_pad, std::vector<float>& output) {
    int64_t output_height = n_out;
    int64_t kernel_w = n_win;
    int64_t stride_w = n_hop;
    int64_t width = n_out;

    output.resize(width, 0.0f);

    int64_t col_idx = 0;
    for (int64_t w_col = 0; w_col < width; ++w_col) {
        int64_t start = w_col * stride_w - n_pad;
        int64_t end = start + kernel_w;
        for (int64_t w_im = start; w_im < end; ++w_im) {
            if (w_im >= 0 && w_im < output_height && col_idx < static_cast<int64_t>(data.size())) {
                output[w_im] += data[col_idx];
            }
            col_idx++;
        }
    }
    output.resize(n_out - 2 * n_pad);
}

std::vector<float> embd_to_audio(const float* embd, int n_codes, int n_embd, int n_thread) {
    const int n_fft = 1280;
    const int n_hop = 320;
    const int n_win = 1280;
    const int n_pad = (n_win - n_hop) / 2;
    const int n_out = (n_codes - 1) * n_hop + n_win;

    std::vector<float> hann(n_fft);
    fill_hann_window(hann.size(), true, hann.data());

    int n_spec = n_embd * n_codes;
    std::vector<float> E(n_spec), S(n_spec), ST(n_spec);

    for (int l = 0; l < n_codes; ++l) {
        for (int k = 0; k < n_embd; ++k) {
            E[k * n_codes + l] = embd[l * n_embd + k];
        }
    }

    for (int k = 0; k < n_embd / 2; ++k) {
        for (int l = 0; l < n_codes; ++l) {
            float mag = E[(k) * n_codes + l];
            float phi = E[(k + n_embd / 2) * n_codes + l];
            mag = exp(mag);
            if (mag > 1e2f) mag = 1e2f;
            S[2 * (k * n_codes + l) + 0] = mag * cosf(phi);
            S[2 * (k * n_codes + l) + 1] = mag * sinf(phi);
        }
    }

    for (int l = 0; l < n_codes; ++l) {
        for (int k = 0; k < n_embd / 2; ++k) {
            ST[l * n_embd + 2 * k + 0] = S[2 * (k * n_codes + l) + 0];
            ST[l * n_embd + 2 * k + 1] = S[2 * (k * n_codes + l) + 1];
        }
    }

    std::vector<float> res(n_codes * n_fft);
    std::vector<float> hann2(n_codes * n_fft);

    std::vector<std::thread> workers(n_thread);
    for (int i = 0; i < n_thread; ++i) {
        workers[i] = std::thread([&, i]() {
            for (int l = i; l < n_codes; l += n_thread) {
                irfft(n_fft, ST.data() + l * n_embd, res.data() + l * n_fft);
                for (int j = 0; j < n_fft; ++j) {
                    res[l * n_fft + j] *= hann[j];
                    hann2[l * n_fft + j] = hann[j] * hann[j];
                }
            }
        });
    }
    for (int i = 0; i < n_thread; ++i) {
        workers[i].join();
    }

    std::vector<float> audio, env;
    fold(res, n_out, n_win, n_hop, n_pad, audio);
    fold(hann2, n_out, n_win, n_hop, n_pad, env);

    for (size_t i = 0; i < audio.size(); ++i) {
        audio[i] /= env[i];
    }
    return audio;
}

// ============================================================================
// Text preprocessing (from tts.cpp)
// ============================================================================

static const std::map<int, std::string> ones = {
    {0, "zero"}, {1, "one"}, {2, "two"}, {3, "three"}, {4, "four"},
    {5, "five"}, {6, "six"}, {7, "seven"}, {8, "eight"}, {9, "nine"},
    {10, "ten"}, {11, "eleven"}, {12, "twelve"}, {13, "thirteen"}, {14, "fourteen"},
    {15, "fifteen"}, {16, "sixteen"}, {17, "seventeen"}, {18, "eighteen"}, {19, "nineteen"}
};

static const std::map<int, std::string> tens = {
    {2, "twenty"}, {3, "thirty"}, {4, "forty"}, {5, "fifty"},
    {6, "sixty"}, {7, "seventy"}, {8, "eighty"}, {9, "ninety"}
};

std::string convert_less_than_thousand(int num) {
    std::string result;
    if (num >= 100) {
        result += ones.at(num / 100) + " hundred ";
        num %= 100;
    }
    if (num >= 20) {
        result += tens.at(num / 10);
        if (num % 10 > 0) result += "-" + ones.at(num % 10);
    } else if (num > 0) {
        result += ones.at(num);
    }
    return result;
}

std::string number_to_words(const std::string& number_str) {
    try {
        size_t decimal_pos = number_str.find('.');
        std::string integer_part = number_str.substr(0, decimal_pos);
        int int_number = std::stoi(integer_part);
        std::string result;

        if (int_number == 0) {
            result = "zero";
        } else {
            if (int_number >= 1000000000) {
                result += convert_less_than_thousand(int_number / 1000000000) + " billion ";
                int_number %= 1000000000;
            }
            if (int_number >= 1000000) {
                result += convert_less_than_thousand(int_number / 1000000) + " million ";
                int_number %= 1000000;
            }
            if (int_number >= 1000) {
                result += convert_less_than_thousand(int_number / 1000) + " thousand ";
                int_number %= 1000;
            }
            if (int_number > 0) {
                result += convert_less_than_thousand(int_number);
            }
        }

        if (decimal_pos != std::string::npos) {
            result += " point";
            std::string decimal_part = number_str.substr(decimal_pos + 1);
            for (char digit : decimal_part) {
                result += " " + ones.at(digit - '0');
            }
        }
        return result;
    } catch (...) {
        return " ";
    }
}

std::string process_text(const std::string& text) {
    // Replace numbers with words
    std::regex number_pattern(R"(\d+(\.\d+)?)");
    std::string processed;
    auto it = std::sregex_iterator(text.begin(), text.end(), number_pattern);
    auto end = std::sregex_iterator();
    size_t last_pos = 0;
    for (auto i = it; i != end; ++i) {
        const std::smatch& match = *i;
        processed.append(text, last_pos, match.position() - last_pos);
        processed.append(number_to_words(match.str()));
        last_pos = match.position() + match.length();
    }
    processed.append(text, last_pos);

    // Lowercase (safe cast for non-ASCII bytes)
    for (auto& c : processed) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    // Normalize punctuation to spaces
    processed = std::regex_replace(processed, std::regex(R"([-_/,\.\\])"), " ");
    // Remove non-alpha
    processed = std::regex_replace(processed, std::regex(R"([^a-z\s])"), "");
    // Collapse whitespace
    processed = std::regex_replace(processed, std::regex(R"(\s+)"), " ");
    // Trim
    processed = std::regex_replace(processed, std::regex(R"(^\s+|\s+$)"), "");
    // Replace spaces with separator tokens
    processed = std::regex_replace(processed, std::regex(R"(\s)"), "<|text_sep|>");

    return processed;
}

std::string process_text_v3(const std::string& text) {
    // Replace numbers with words
    std::regex number_pattern(R"(\d+(\.\d+)?)");
    std::string processed;
    auto it = std::sregex_iterator(text.begin(), text.end(), number_pattern);
    auto end = std::sregex_iterator();
    size_t last_pos = 0;
    for (auto i = it; i != end; ++i) {
        const std::smatch& match = *i;
        processed.append(text, last_pos, match.position() - last_pos);
        processed.append(number_to_words(match.str()));
        last_pos = match.position() + match.length();
    }
    processed.append(text, last_pos);

    // Lowercase (safe cast for non-ASCII bytes)
    for (auto& c : processed) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    // Replace punctuation with special tokens (before stripping)
    processed = std::regex_replace(processed, std::regex(R"(\.)"), "<|period|>");
    processed = std::regex_replace(processed, std::regex(R"(,)"), "<|comma|>");
    processed = std::regex_replace(processed, std::regex(R"(\?)"), "<|question_mark|>");
    processed = std::regex_replace(processed, std::regex(R"(!)"), "<|exclamation_mark|>");

    // Strip unwanted ASCII punctuation/control chars, but preserve:
    //   - letters (a-z), digits (for any stragglers), spaces
    //   - non-ASCII bytes (UTF-8 multibyte sequences for multilingual)
    //   - special tokens <|...|>
    std::string cleaned;
    size_t pos = 0;
    while (pos < processed.size()) {
        auto uc = static_cast<unsigned char>(processed[pos]);
        // Preserve special tokens
        if (processed[pos] == '<' && pos + 1 < processed.size() && processed[pos + 1] == '|') {
            size_t close = processed.find("|>", pos + 2);
            if (close != std::string::npos) {
                cleaned += processed.substr(pos, close + 2 - pos);
                pos = close + 2;
                continue;
            }
        }
        // Keep non-ASCII (UTF-8 continuation/lead bytes)
        if (uc > 127) {
            cleaned += processed[pos];
        }
        // Keep ASCII letters, digits, spaces
        else if ((uc >= 'a' && uc <= 'z') || (uc >= '0' && uc <= '9') || uc == ' ') {
            cleaned += processed[pos];
        }
        // Drop everything else (ASCII punctuation, control chars)
        ++pos;
    }
    processed = cleaned;

    // Collapse whitespace and trim
    processed = std::regex_replace(processed, std::regex(R"(\s+)"), " ");
    processed = std::regex_replace(processed, std::regex(R"(^\s+|\s+$)"), "");

    // Replace spaces with <|space|>
    processed = std::regex_replace(processed, std::regex(R"( )"), "<|space|>");

    return processed;
}

// ============================================================================
// Default speaker profile (OuteTTS v0.2 en_male_1)
// Token IDs pre-computed from the speaker JSON to avoid runtime tokenization.
// ============================================================================

const std::string default_audio_text =
    "<|text_start|>the<|text_sep|>overall<|text_sep|>package<|text_sep|>from<|text_sep|>"
    "just<|text_sep|>two<|text_sep|>people<|text_sep|>is<|text_sep|>pretty<|text_sep|>"
    "remarkable<|text_sep|>sure<|text_sep|>i<|text_sep|>have<|text_sep|>some<|text_sep|>"
    "critiques<|text_sep|>about<|text_sep|>some<|text_sep|>of<|text_sep|>the<|text_sep|>"
    "gameplay<|text_sep|>aspects<|text_sep|>but<|text_sep|>its<|text_sep|>still<|text_sep|>"
    "really<|text_sep|>enjoyable<|text_sep|>and<|text_sep|>it<|text_sep|>looks<|text_sep|>"
    "lovely<|text_sep|>";

const std::string default_audio_data =
    R"(<|audio_start|>
the<|t_0.08|><|code_start|><|257|><|740|><|636|><|913|><|788|><|1703|><|code_end|>
overall<|t_0.36|><|code_start|><|127|><|201|><|191|><|774|><|700|><|532|><|1056|><|557|><|798|><|298|><|1741|><|747|><|1662|><|1617|><|1702|><|1527|><|368|><|1588|><|1049|><|1008|><|1625|><|747|><|1576|><|728|><|1019|><|1696|><|1765|><|code_end|>
package<|t_0.56|><|code_start|><|935|><|584|><|1319|><|627|><|1016|><|1491|><|1344|><|1117|><|1526|><|1040|><|239|><|1435|><|951|><|498|><|723|><|1180|><|535|><|789|><|1649|><|1637|><|78|><|465|><|1668|><|901|><|595|><|1675|><|117|><|1009|><|1667|><|320|><|840|><|79|><|507|><|1762|><|1508|><|1228|><|1768|><|802|><|1450|><|1457|><|232|><|639|><|code_end|>
from<|t_0.19|><|code_start|><|604|><|782|><|1682|><|872|><|1532|><|1600|><|1036|><|1761|><|647|><|1554|><|1371|><|653|><|1595|><|950|><|code_end|>
just<|t_0.25|><|code_start|><|1782|><|1670|><|317|><|786|><|1748|><|631|><|599|><|1155|><|1364|><|1524|><|36|><|1591|><|889|><|1535|><|541|><|440|><|1532|><|50|><|870|><|code_end|>
two<|t_0.24|><|code_start|><|1681|><|1510|><|673|><|799|><|805|><|1342|><|330|><|519|><|62|><|640|><|1138|><|565|><|1552|><|1497|><|1552|><|572|><|1715|><|1732|><|code_end|>
people<|t_0.39|><|code_start|><|593|><|274|><|136|><|740|><|691|><|633|><|1484|><|1061|><|1138|><|1485|><|344|><|428|><|397|><|1562|><|645|><|917|><|1035|><|1449|><|1669|><|487|><|442|><|1484|><|1329|><|1832|><|1704|><|600|><|761|><|653|><|269|><|code_end|>
is<|t_0.16|><|code_start|><|566|><|583|><|1755|><|646|><|1337|><|709|><|802|><|1008|><|485|><|1583|><|652|><|10|><|code_end|>
pretty<|t_0.32|><|code_start|><|1818|><|1747|><|692|><|733|><|1010|><|534|><|406|><|1697|><|1053|><|1521|><|1355|><|1274|><|816|><|1398|><|211|><|1218|><|817|><|1472|><|1703|><|686|><|13|><|822|><|445|><|1068|><|code_end|>
remarkable<|t_0.68|><|code_start|><|230|><|1048|><|1705|><|355|><|706|><|1149|><|1535|><|1787|><|1356|><|1396|><|835|><|1583|><|486|><|1249|><|286|><|937|><|1076|><|1150|><|614|><|42|><|1058|><|705|><|681|><|798|><|934|><|490|><|514|><|1399|><|572|><|1446|><|1703|><|1346|><|1040|><|1426|><|1304|><|664|><|171|><|1530|><|625|><|64|><|1708|><|1830|><|1030|><|443|><|1509|><|1063|><|1605|><|1785|><|721|><|1440|><|923|><|code_end|>
sure<|t_0.36|><|code_start|><|792|><|1780|><|923|><|1640|><|265|><|261|><|1525|><|567|><|1491|><|1250|><|1730|><|362|><|919|><|1766|><|543|><|1|><|333|><|113|><|970|><|252|><|1606|><|133|><|302|><|1810|><|1046|><|1190|><|1675|><|code_end|>
i<|t_0.08|><|code_start|><|123|><|439|><|1074|><|705|><|1799|><|637|><|code_end|>
have<|t_0.16|><|code_start|><|1509|><|599|><|518|><|1170|><|552|><|1029|><|1267|><|864|><|419|><|143|><|1061|><|0|><|code_end|>
some<|t_0.16|><|code_start|><|619|><|400|><|1270|><|62|><|1370|><|1832|><|917|><|1661|><|167|><|269|><|1366|><|1508|><|code_end|>
critiques<|t_0.60|><|code_start|><|559|><|584|><|1163|><|1129|><|1313|><|1728|><|721|><|1146|><|1093|><|577|><|928|><|27|><|630|><|1080|><|1346|><|1337|><|320|><|1382|><|1175|><|1682|><|1556|><|990|><|1683|><|860|><|1721|><|110|><|786|><|376|><|1085|><|756|><|1523|><|234|><|1334|><|1506|><|1578|><|659|><|612|><|1108|><|1466|><|1647|><|308|><|1470|><|746|><|556|><|1061|><|code_end|>
about<|t_0.29|><|code_start|><|26|><|1649|><|545|><|1367|><|1263|><|1728|><|450|><|859|><|1434|><|497|><|1220|><|1285|><|179|><|755|><|1154|><|779|><|179|><|1229|><|1213|><|922|><|1774|><|1408|><|code_end|>
some<|t_0.23|><|code_start|><|986|><|28|><|1649|><|778|><|858|><|1519|><|1|><|18|><|26|><|1042|><|1174|><|1309|><|1499|><|1712|><|1692|><|1516|><|1574|><|code_end|>
of<|t_0.07|><|code_start|><|197|><|716|><|1039|><|1662|><|64|><|code_end|>
the<|t_0.08|><|code_start|><|1811|><|1568|><|569|><|886|><|1025|><|1374|><|code_end|>
gameplay<|t_0.48|><|code_start|><|1269|><|1092|><|933|><|1362|><|1762|><|1700|><|1675|><|215|><|781|><|1086|><|461|><|838|><|1022|><|759|><|649|><|1416|><|1004|><|551|><|909|><|787|><|343|><|830|><|1391|><|1040|><|1622|><|1779|><|1360|><|1231|><|1187|><|1317|><|76|><|997|><|989|><|978|><|737|><|189|><|code_end|>
aspects<|t_0.56|><|code_start|><|1423|><|797|><|1316|><|1222|><|147|><|719|><|1347|><|386|><|1390|><|1558|><|154|><|440|><|634|><|592|><|1097|><|1718|><|712|><|763|><|1118|><|1721|><|1311|><|868|><|580|><|362|><|1435|><|868|><|247|><|221|><|886|><|1145|><|1274|><|1284|><|457|><|1043|><|1459|><|1818|><|62|><|599|><|1035|><|62|><|1649|><|778|><|code_end|>
but<|t_0.20|><|code_start|><|780|><|1825|><|1681|><|1007|><|861|><|710|><|702|><|939|><|1669|><|1491|><|613|><|1739|><|823|><|1469|><|648|><|code_end|>
its<|t_0.09|><|code_start|><|92|><|688|><|1623|><|962|><|1670|><|527|><|599|><|code_end|>
still<|t_0.27|><|code_start|><|636|><|10|><|1217|><|344|><|713|><|957|><|823|><|154|><|1649|><|1286|><|508|><|214|><|1760|><|1250|><|456|><|1352|><|1368|><|921|><|615|><|5|><|code_end|>
really<|t_0.36|><|code_start|><|55|><|420|><|1008|><|1659|><|27|><|644|><|1266|><|617|><|761|><|1712|><|109|><|1465|><|1587|><|503|><|1541|><|619|><|197|><|1019|><|817|><|269|><|377|><|362|><|1381|><|507|><|1488|><|4|><|1695|><|code_end|>
enjoyable<|t_0.49|><|code_start|><|678|><|501|><|864|><|319|><|288|><|1472|><|1341|><|686|><|562|><|1463|><|619|><|1563|><|471|><|911|><|730|><|1811|><|1006|><|520|><|861|><|1274|><|125|><|1431|><|638|><|621|><|153|><|876|><|1770|><|437|><|987|><|1653|><|1109|><|898|><|1285|><|80|><|593|><|1709|><|843|><|code_end|>
and<|t_0.15|><|code_start|><|1285|><|987|><|303|><|1037|><|730|><|1164|><|502|><|120|><|1737|><|1655|><|1318|><|code_end|>
it<|t_0.09|><|code_start|><|848|><|1366|><|395|><|1601|><|1513|><|593|><|1302|><|code_end|>
looks<|t_0.27|><|code_start|><|1281|><|1266|><|1755|><|572|><|248|><|1751|><|1257|><|695|><|1380|><|457|><|659|><|585|><|1315|><|1105|><|1776|><|736|><|24|><|736|><|654|><|1027|><|code_end|>
lovely<|t_0.56|><|code_start|><|634|><|596|><|1766|><|1556|><|1306|><|1285|><|1481|><|1721|><|1123|><|438|><|1246|><|1251|><|795|><|659|><|1381|><|1658|><|217|><|1772|><|562|><|952|><|107|><|1129|><|1112|><|467|><|550|><|1079|><|840|><|1615|><|1469|><|1380|><|168|><|917|><|836|><|1827|><|437|><|583|><|67|><|595|><|1087|><|1646|><|1493|><|1677|><|code_end|>)";

// llama.cpp log filter (reuse pattern from runner.cpp)
void filtered_llama_log(enum ggml_log_level level, const char* text, void* /*user_data*/) {
    if (!text || text[0] == '.' || text[0] == '\n' || text[0] == '\0') return;
    static int filter_level = -1;
    if (filter_level == -1) {
        const char* env = std::getenv("LLAMA_LOG_LEVEL");
        filter_level = env ?
            (std::string(env) == "info" ? GGML_LOG_LEVEL_INFO :
             std::string(env) == "warn" ? GGML_LOG_LEVEL_WARN :
             std::string(env) == "debug" ? GGML_LOG_LEVEL_DEBUG :
             GGML_LOG_LEVEL_ERROR) : GGML_LOG_LEVEL_ERROR;
    }
    if (level >= filter_level) {
        fprintf(stderr, "%s", text);
    }
}

} // anonymous namespace

// ============================================================================
// TtsRunner implementation
// ============================================================================

TtsRunner::TtsRunner(const std::string& modelPath, const std::string& vocoderPath) {
    llama_log_set(filtered_llama_log, nullptr);
    ggml_backend_load_all();

    std::lock_guard<std::mutex> lock(tts_model_mutex_);

    // Load TTS model (OuteTTS)
    if (!shared_tts_model_ || current_tts_model_path_ != modelPath) {
        LOG_INFO("Loading TTS model: " + modelPath);
        llama_model_params model_params = llama_model_default_params();
#if defined(__APPLE__)
        model_params.n_gpu_layers = env_int("NRVNA_GPU_LAYERS", 99);
#else
        model_params.n_gpu_layers = env_int("NRVNA_GPU_LAYERS", 0);
#endif
        llama_model* model = llama_model_load_from_file(modelPath.c_str(), model_params);
        if (!model) {
            throw std::runtime_error("Failed to load TTS model: " + modelPath);
        }
        shared_tts_model_ = std::shared_ptr<llama_model>(model, llama_model_free);
        current_tts_model_path_ = modelPath;
        LOG_INFO("TTS model loaded");

        // Detect OuteTTS version by probing for v0.3 special token <|space|>
        const llama_vocab* det_vocab = llama_model_get_vocab(shared_tts_model_.get());
        {
            llama_token buf[4];
            int n = llama_tokenize(det_vocab, "<|space|>", 9, buf, 4, false, true);
            if (n == 1) {
                char piece[64] = {};
                int plen = llama_token_to_piece(det_vocab, buf[0], piece, sizeof(piece), 0, true);
                if (plen > 0 && std::string(piece, plen) == "<|space|>") {
                    detected_version_ = TtsVersion::V0_3;
                    LOG_INFO("Detected OuteTTS v0.3");
                } else {
                    detected_version_ = TtsVersion::V0_2;
                    LOG_INFO("Detected OuteTTS v0.2");
                }
            } else {
                detected_version_ = TtsVersion::V0_2;
                LOG_INFO("Detected OuteTTS v0.2");
            }
        }

        // Cache v0.3 speaker transforms
        if (detected_version_ == TtsVersion::V0_3) {
            // Transform speaker text: replace <|text_sep|> with <|space|>
            v3_audio_text_ = default_audio_text;
            {
                const std::string from = "<|text_sep|>";
                const std::string to = "<|space|>";
                size_t pos = 0;
                while ((pos = v3_audio_text_.find(from, pos)) != std::string::npos) {
                    v3_audio_text_.replace(pos, from.size(), to);
                    pos += to.size();
                }
                // Strip <|text_start|> prefix if present
                const std::string ts = "<|text_start|>";
                if (v3_audio_text_.substr(0, ts.size()) == ts) {
                    v3_audio_text_.erase(0, ts.size());
                }
            }

            // Transform speaker audio data: strip code_start/code_end, join with <|space|>\n
            v3_audio_data_.clear();
            {
                std::string src = default_audio_data;
                // Strip outer <|audio_start|> and trailing whitespace
                const std::string as = "<|audio_start|>";
                size_t as_pos = src.find(as);
                if (as_pos != std::string::npos) {
                    src.erase(0, as_pos + as.size());
                }
                // Process line by line
                size_t pos = 0;
                bool first = true;
                while (pos < src.size()) {
                    size_t nl = src.find('\n', pos);
                    if (nl == std::string::npos) nl = src.size();
                    std::string line = src.substr(pos, nl - pos);
                    pos = nl + 1;

                    // Skip empty lines
                    if (line.empty() || line.find_first_not_of(" \t\r") == std::string::npos) continue;

                    // Strip <|code_start|> and <|code_end|>
                    const std::string cs = "<|code_start|>";
                    const std::string ce = "<|code_end|>";
                    size_t cs_pos = line.find(cs);
                    if (cs_pos != std::string::npos) line.erase(cs_pos, cs.size());
                    size_t ce_pos = line.find(ce);
                    if (ce_pos != std::string::npos) line.erase(ce_pos, ce.size());

                    if (!first) {
                        v3_audio_data_ += "<|space|>\n";
                    }
                    v3_audio_data_ += line;
                    first = false;
                }
            }
            LOG_DEBUG("v0.3 speaker text cached (" + std::to_string(v3_audio_text_.size()) + " chars)");
            LOG_DEBUG("v0.3 speaker data cached (" + std::to_string(v3_audio_data_.size()) + " chars)");
        }

        // Validate model has essential TTS tokens — fail fast at startup
        {
            auto probe = [&](const char* token_str) -> bool {
                llama_token buf[4];
                int n = llama_tokenize(det_vocab, token_str,
                    static_cast<int32_t>(strlen(token_str)), buf, 4, false, true);
                return n == 1;
            };
            bool ok = probe("<|im_start|>") && probe("<|text_start|>")
                   && probe("<|text_end|>") && probe("<|audio_start|>");
            if (!ok) {
                throw std::runtime_error(
                    "Model is not TTS-compatible (missing required special tokens): " + modelPath);
            }
        }
    }

    // Load vocoder (WavTokenizer)
    if (!shared_vocoder_ || current_vocoder_path_ != vocoderPath) {
        LOG_INFO("Loading vocoder: " + vocoderPath);
        llama_model_params model_params = llama_model_default_params();
        // Vocoder is tiny (~50MB) and llama_encode with embeddings mode
        // fails on Metal without compute (pre-M1 Macs). CPU-only is fine.
        model_params.n_gpu_layers = 0;
        llama_model* model = llama_model_load_from_file(vocoderPath.c_str(), model_params);
        if (!model) {
            throw std::runtime_error("Failed to load vocoder: " + vocoderPath);
        }
        shared_vocoder_ = std::shared_ptr<llama_model>(model, llama_model_free);
        current_vocoder_path_ = vocoderPath;
        LOG_INFO("Vocoder loaded");
    }
}

TtsRunner::~TtsRunner() = default;

TtsResult TtsRunner::run(const std::string& text) {
    if (!shared_tts_model_ || !shared_vocoder_) {
        return {false, {}, 24000, "TTS models not loaded"};
    }

    try {
        const llama_vocab* vocab = llama_model_get_vocab(shared_tts_model_.get());

        std::string full_prompt;
        if (detected_version_ == TtsVersion::V0_3) {
            std::string processed = process_text_v3(text);
            LOG_DEBUG("TTS v0.3 processed text: " + processed);
            full_prompt = "<|im_start|>\n<|text_start|>" + v3_audio_text_
                + "<|space|>" + processed
                + "<|text_end|>\n<|audio_start|>\n" + v3_audio_data_
                + "<|space|>\n";
        } else {
            std::string processed = process_text(text);
            LOG_DEBUG("TTS v0.2 processed text: " + processed);
            full_prompt = "<|im_start|>\n" + default_audio_text + processed
                + "<|text_end|>\n" + default_audio_data;
        }

        // Tokenize
        int n_tokens = -llama_tokenize(vocab, full_prompt.c_str(), full_prompt.size(), nullptr, 0, true, true);
        if (n_tokens <= 0) {
            return {false, {}, 24000, "Failed to tokenize TTS prompt"};
        }

        std::vector<llama_token> prompt_tokens(n_tokens);
        if (llama_tokenize(vocab, full_prompt.c_str(), full_prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
            return {false, {}, 24000, "Failed to tokenize TTS prompt"};
        }

        LOG_INFO("TTS prompt: " + std::to_string(prompt_tokens.size()) + " tokens");

        // Create context for text-to-codes generation
        int n_predict = env_int("NRVNA_PREDICT", 4096);
        int n_ctx_train = llama_model_n_ctx_train(shared_tts_model_.get());
        int max_ctx = std::min(n_ctx_train, env_int("NRVNA_MAX_CTX", 8192));
        int n_prompt = static_cast<int>(prompt_tokens.size());
        int n_ctx = std::min(n_prompt + n_predict, max_ctx);

        if (n_prompt >= n_ctx) {
            return {false, {}, 24000,
                "TTS prompt too long (" + std::to_string(n_prompt) + " tokens, context limit " +
                std::to_string(max_ctx) + "). Try shorter text."};
        }

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = n_ctx;
        ctx_params.n_batch = env_int("NRVNA_BATCH", 8192);
        ctx_params.no_perf = false;

        llama_context* ctx_ttc = llama_init_from_model(shared_tts_model_.get(), ctx_params);
        if (!ctx_ttc) {
            return {false, {}, 24000, "Failed to create TTS context"};
        }

        // Build sampler — TTS uses top_k=4 only
        auto sparams = llama_sampler_chain_default_params();
        sparams.no_perf = false;
        llama_sampler* smpl = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(4));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(env_int("NRVNA_SEED", 0)));

        // Eval prompt
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        if (llama_decode(ctx_ttc, batch) != 0) {
            llama_sampler_free(smpl);
            llama_free(ctx_ttc);
            return {false, {}, 24000, "Failed to decode TTS prompt"};
        }

        // Generate code tokens
        std::vector<llama_token> codes;

        for (int i = 0; i < n_predict; ++i) {
            llama_token new_token = llama_sampler_sample(smpl, ctx_ttc, -1);
            llama_sampler_accept(smpl, new_token);

            if (llama_vocab_is_eog(vocab, new_token)) {
                break;
            }

            codes.push_back(new_token);

            batch = llama_batch_get_one(&new_token, 1);
            if (llama_decode(ctx_ttc, batch) != 0) {
                LOG_WARN("TTS decode failed at token " + std::to_string(i));
                break;
            }
        }

        llama_sampler_free(smpl);
        llama_free(ctx_ttc);

        LOG_INFO("TTS generated " + std::to_string(codes.size()) + " code tokens");

        // Extract audio codes by parsing token text (handles non-contiguous token IDs in v0.3)
        {
            std::vector<llama_token> audio_codes;
            for (auto token : codes) {
                char piece[32] = {};
                int plen = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, true);
                if (plen < 5) continue;
                std::string s(piece, plen);
                // Match <|N|> pattern
                if (s.size() >= 5 && s[0] == '<' && s[1] == '|'
                    && s[s.size()-2] == '|' && s[s.size()-1] == '>') {
                    std::string num_str = s.substr(2, s.size() - 4);
                    // Verify all digits
                    bool all_digits = !num_str.empty();
                    for (char c : num_str) {
                        if (c < '0' || c > '9') { all_digits = false; break; }
                    }
                    if (all_digits) {
                        int code = std::stoi(num_str);
                        if (code >= 0 && code <= 4100) {
                            audio_codes.push_back(static_cast<llama_token>(code));
                        }
                    }
                }
            }
            codes = std::move(audio_codes);
        }

        LOG_INFO("TTS audio tokens after filter: " + std::to_string(codes.size()));

        if (codes.empty()) {
            return {false, {}, 24000, "No audio tokens generated"};
        }

        // Vocoder: encode codes to get embeddings
        int n_codes = static_cast<int>(codes.size());
        llama_context_params voc_params = llama_context_default_params();
        voc_params.n_ctx = n_codes;
        voc_params.n_batch = n_codes;
        voc_params.embeddings = true;

        llama_context* ctx_voc = llama_init_from_model(shared_vocoder_.get(), voc_params);
        if (!ctx_voc) {
            return {false, {}, 24000, "Failed to create vocoder context"};
        }

        llama_batch voc_batch = llama_batch_init(n_codes, 0, 1);
        for (int i = 0; i < n_codes; ++i) {
            voc_batch.token[i] = codes[i];
            voc_batch.pos[i] = i;
            voc_batch.n_seq_id[i] = 1;
            voc_batch.seq_id[i][0] = 0;
            voc_batch.logits[i] = true;
        }
        voc_batch.n_tokens = n_codes;

        if (llama_encode(ctx_voc, voc_batch) != 0) {
            llama_batch_free(voc_batch);
            llama_free(ctx_voc);
            return {false, {}, 24000, "Vocoder encode failed"};
        }

        llama_batch_free(voc_batch);

        // Get embeddings and convert to audio
        int n_embd = llama_model_n_embd(shared_vocoder_.get());
        const float* embd = llama_get_embeddings(ctx_voc);
        if (!embd) {
            llama_free(ctx_voc);
            return {false, {}, 24000, "Failed to get vocoder embeddings"};
        }

        int n_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
        auto audio = embd_to_audio(embd, n_codes, n_embd, n_threads);

        llama_free(ctx_voc);

        // Zero out first 0.25 seconds (artifact suppression, from tts.cpp)
        int silence_samples = std::min(static_cast<int>(audio.size()), 24000 / 4);
        for (int i = 0; i < silence_samples; ++i) {
            audio[i] = 0.0f;
        }

        LOG_INFO("TTS generated " + std::to_string(audio.size()) + " audio samples");

        return {true, std::move(audio), 24000, ""};

    } catch (const std::exception& e) {
        LOG_ERROR("TTS error: " + std::string(e.what()));
        return {false, {}, 24000, "TTS error: " + std::string(e.what())};
    }
}

} // namespace nrvnaai
