# Aneja Lab
# AAA Volumetrics
# Aneja Lab | Yale School of Medicine
# Thomas Hager
# Created (12/09/2024)
# Updated (12/09/2024)

"""
A collection of scripts designed to return a variety of different metrics 
from model inferences. These scripts are designed to be run individually and
not as one cohesive file.
"""

import numpy as np
import os
import SimpleITK as sitk
from tqdm import tqdm
import pandas as pd
import nibabel as nib

# Function to return the dice score from two images
def myDice(img1, img2):
    intersection = np.logical_and(img1, img2)
    union = np.logical_or(img1, img2)
    dice = (2*np.sum(intersection))/(np.sum(union)+np.sum(intersection))
    return dice


# Script designed to return the dice score from manual and automatic masks
results = []
workingDir = "/path/to/directory"
fileList = os.listdir(workingDir)

for file in tqdm(fileList, desc="Progress"):
    if file.endswith("_0000.nii.gz") or file.endswith("Manual.nii.gz"):
        continue

    if file.endswith(".json"):
        continue

    automaticPath = os.path.join(workingDir, file)
    manualPath = os.path.join(workingDir, file[:-7] + "_Manual.nii.gz")
    fileName = file[:-7]

    automaticImage = sitk.ReadImage(automaticPath)
    automaticArray = sitk.GetArrayFromImage(automaticImage)

    manualImage = sitk.ReadImage(manualPath)
    manualArray = sitk.GetArrayFromImage(manualImage)

    dice = myDice(automaticArray, manualArray)

    print(f"Patient: {fileName}, Dice: {dice}")
    results.append([fileName, dice])

pd.DataFrame(results).to_csv("/path/to/csv/destination/name.csv",
                             header=["Case", "Dice Score"],
                             index=False)

res = pd.read_csv("/path/to/csv/destination/name.csv")
print(res)

# A script designed return the Hausdorff distance of a manual and automatic mask
results = []
workingDir = "/path/to/directory"
saveName = "LUHaus.csv"
savePath = os.path.join(workingDir, saveName)
fileList = os.listdir(workingDir)

for file in tqdm(fileList, desc="Progress"):
    if file.endswith("_0000.nii.gz") or file.endswith("Manual.nii.gz"):
        continue

    if file.endswith(".json"):
        continue

    try:
        automaticPath = os.path.join(workingDir, file)
        manualPath = os.path.join(workingDir, file[:-7] + "_Manual.nii.gz")
        fileName = file[:-7]

        inImg = sitk.ReadImage(automaticPath)
        inImg = sitk.GetArrayFromImage(inImg)
        refImg = sitk.ReadImage(manualPath)
        refImg = sitk.GetArrayFromImage(refImg)

        print("\n" + str(np.shape(inImg)))
        print("\n" + str(np.shape(refImg)))
        sum = np.sum(inImg)
        if (sum == 0):
            continue
        refContourImg = sitk.GetImageFromArray(inImg)
        testContourImg = sitk.GetImageFromArray(refImg)

        referenceSegmentation = sitk.Cast(refContourImg, sitk.sitkUInt32)
        seg = sitk.Cast(testContourImg, sitk.sitkUInt32)

        referenceSurface = sitk.LabelContour(referenceSegmentation, False)
        segSurface = sitk.LabelContour(seg, False)

        segDistanceMap = sitk.Abs(
            sitk.SignedMaurerDistanceMap(segSurface,
                                         squaredDistance=False,
                                         useImageSpacing=True))

        referenceSegmentationDistanceMap = sitk.Abs(
            sitk.SignedMaurerDistanceMap(referenceSegmentation,
                                         squaredDistance=False,
                                         useImageSpacing=True))

        distSeg = sitk.GetArrayViewFromImage(segDistanceMap)[
            sitk.GetArrayViewFromImage(referenceSurface) == 1]
        distRef = sitk.GetArrayViewFromImage(referenceSegmentationDistanceMap)[
            sitk.GetArrayViewFromImage(segSurface) == 1]

        print("95%")
        haus = (np.percentile(distRef, 95) +
                np.percentile(distSeg, 95)) / 2.0
        print(haus)

        print(f"Patient: {fileName}, Haus: {haus}")
        results.append([fileName, haus])

    except Exception as e:
        print(f"Error {e} with {fileName}")

pd.DataFrame(results).to_csv(savePath,
                             header=["Case", "Haus"],
                             index=False)

res = pd.read_csv(savePath)
print(res)

# A script to return the Jaccard's Index of a manual and automatic mask
baseDir = "/path/to/base/directory"
inferenceList = os.listdir(baseDir)

for inf in tqdm(inferenceList, desc = "Inference Progress"):
    inferenceDir = os.path.join(baseDir, inf)
    niftiList = os.listdir(inferenceDir)

    results = []
    for nf in tqdm(niftiList, desc = "Scan Progress"):
        if nf.endswith(".json") or nf.endswith(".csv"):
            continue

        manual = ""
        automatic = ""
        if nf.endswith("0000.nii.gz"):
            manual = nf[:-11] + "Manual.nii.gz"
            automatic = nf[:-12] + ".nii.gz"

            manualNib = nib.load(os.path.join(inferenceDir, manual))
            automaticNib = nib.load(os.path.join(inferenceDir, automatic))

            manualArray = manualNib.get_fdata()
            automaticArray = automaticNib.get_fdata()

            manualArray = manualArray.flatten()
            automaticArray = automaticArray.flatten()

            JI = jaccard_score(manualArray, automaticArray, average="micro")
            results.append([nf[:-12], JI])

        else:
            continue

    resultDF = pd.DataFrame(results)
    print(resultDF)
    resultDF.to_csv(os.path.join(inferenceDir, "JIScore.csv"))

# Desgined to return the voxel spacing of a set of images
root = "/path/to/root"
savePath = os.path.join("/base/path", "fileName.csv")
studyList = os.listdir(root)
spacingList = []
for image in studyList:
    if image.endswith("AN_Trimmed.nii.gz"):
        continue
    imageRead = sitk.ReadImage(os.path.join(root, image))
    spacing = imageRead.GetSpacing()
    print(spacing)
    spacingList.append(spacing)

pd.DataFrame(spacingList).to_csv(savePath,
                                 header=["xSpacing", "ySpacing", "zSpacing"],
                                 index=False)
