#################################################################################
# Test 03: Offline evaluation -> `compareRes2Ref()` (CPU-Only)
#################################################################################


from pyppbox.utils.evatools import compareRes2Ref


res_txt = "../examples/data/result_gta.mp4.txt"
ref_txt = "../examples/data/gta.mp4.txt"

results = compareRes2Ref(
    res_txt=res_txt, 
    ref_txt=ref_txt, 
    res_box_xyxy_index=5, # 4 is index of the bounding box xyxy in res_txt
    ref_box_xyxy_index=4, # 4 is index of the bounding box xyxy in ref_txt
    res_compare_index=2, # 2 is Name ID in res_txt
    ref_compare_index=2, # 2 is Name ID in ref_txt
    box_max_spread= 5 # Similar to IOU, but it uses distancing
)

with open("test_03/results.txt", 'w') as results_txt:
    for res in results:
        results_txt.write(str(res))
