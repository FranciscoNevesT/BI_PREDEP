from sbibm.algorithms.sbi.mcabc import run as mcabc
from sbibm.algorithms.sbi.smcabc import run as smcabc
from sbibm.algorithms.sbi.snle import run as snle
from sbibm.algorithms.sbi.snpe import run as snpe
from sbibm.algorithms.sbi.snre import run as snre
from sbibm.algorithms.amortized.amortized import run as amortized

rej_abc = mcabc
smc_abc = smcabc
amortized_nn = amortized