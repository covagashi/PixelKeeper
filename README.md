Random scripts.

auto_enhance.py -- use ImageMagick for quick auto enhance
getLowQualityScore.py -- create an json file with the low score pictures from photoprism
imageEnhancementV2_HDR.py -- create HDR effect using CLAHE (Contrast Limited Adaptive Histogram Equalization) using exposure.equalize_adapthist()
noise.py --- remove the noise from pictures using adversarial noise. adjust the variables DIAMETER = 5, SIGMA_COLOR = 8,SIGMA_SPACE = 8, RADIUS = 4, EPS = 16, BILATERAL_ITERATIONS = 64, GUIDED_ITERATIONS = 4
noiseAuto.py -- same as noise but adjusting itself the variables, it's not working at all well, I guess adding face detection with mask it will improve.

