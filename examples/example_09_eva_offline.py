#################################################################################
# Example of using pyppbox's offline evaluation -> `compareRes2Ref()`
#################################################################################

from pyppbox.utils.evatools import compareRes2Ref


# Assuming you already have the result file "result_gta.mp4.txt" from example 07.
# -> Check example 07 if you haven't already ;)
res_txt = "data/result_gta.mp4.txt"
ref_txt = "data/gta.mp4.txt" # The "gta.mp4.txt" is ground-truth for "gta.mp4"

wrong_id, missed_det, fault_det, total_det, score = compareRes2Ref(
    res_txt=res_txt, 
    ref_txt=ref_txt, 
    res_box_xyxy_index=5, # 4 is index of the bounding box xyxy in res_txt
    ref_box_xyxy_index=4, # 4 is index of the bounding box xyxy in ref_txt
    res_compare_index=2, # 2 is Name ID in res_txt
    ref_compare_index=2, # 2 is Name ID in ref_txt
    box_max_spread= 5 # Similar to IOU, but it uses distancing
)

