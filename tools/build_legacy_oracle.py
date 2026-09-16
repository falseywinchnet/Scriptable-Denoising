"""Generate a native-only oracle from the untouched Downloads C++/CLI header.
The separately hosted Cleanup repository is not used. Only wrapper portability
and two empty-input early-return defects are repaired in the generated test copy.
"""
from pathlib import Path
root = Path(__file__).resolve().parents[1]
s = (root/'reference/legacy-plugin/CleanupNative/CleanupNative.h').read_text()
s = s.split('namespace CleanupNative {')[0]
s = s.replace('using namespace System;', '')
s = s.replace('#include <array># define M_PI', '#include <array>\n#define M_PI')
s = '#include <cstdint>\n#include <cstring>\n' + s
s = s.replace('private:', 'public:')
s = s.replace('man = 0.0;\n\t\t}', 'man = 0.0; return;\n\t\t}')
s = s.replace('atd = 0.0;\n\t\t}', 'atd = 0.0; return;\n\t\t}')
out = root/'build/legacy-oracle';out.mkdir(parents=True, exist_ok=True)
(out/'legacy.h').write_text(s)
(out/'oracle.cpp').write_text('''#include "legacy.h"
extern "C" void legacy_mask(const double* input, const double* raw, const double* detected, int nb, int squelch, double* output) {
    Filter f;
    f.NBINS_last=nb;f.silence=squelch!=0;f.mult=1.;f.CONST_last=.057;
    f.generate_true_logistic();f.generate_true_logistic_3();f.determine_entropy_maximum();f.determine_entropy_maximum_3();
    for(int t=0;t<192;++t) {f.entropy_unmasked[t]=raw[t];f.entropy_thresholded[t]=static_cast<int>(detected[t]);
        for(int b=0;b<257;++b)f.stft_real[t][b]=input[t*257+b];}
    f.initial=0.;f.multiplier=0.;f.man_global=0.;f.atd_global=0.;
    f.find_max(f.stft_real,f.initial);f.smooth_and_mask();
    for(int t=0;t<192;++t)for(int b=0;b<257;++b)output[t*257+b]=f.previous[t][b];
}
''')
print(out/'oracle.cpp')
