# Aneja Lab
# AAA Volumetrics
# Aneja Lab | Yale School of Medicine
# Thomas Hager
# Created (12/09/2024)
# Updated (12/09/2024)

"""
A collection of scripts used to preprocess scans before model training.
These scripts are designed to be run individually, not as one cohesive script.
"""

import os
from tqdm import tqdm
import SimpleITK as sitk
import nibabel as nib

# Designed to convert a collection .nrrd files to .nii.gz
root = "/path/to/directory"
studyList = os.listdir(root)

for study in tqdm(studyList, desc="Study Progress"):
    if study.startswith("."):
        continue

    imageList = os.listdir(os.path.join(root, study))
    for image in imageList:
        if image.startswith("."):
            continue

        if image.endswith(".nii.gz"):
            continue

        else:
           img = sitk.ReadImage(os.path.join(root, study, image))
           sitk.WriteImage(img,
                           os.path.join(root,
                                        study,
                                        image[:-5] + ".nii.gz"))

# Designed to normalize scan arrays, from a max of 255 to a max of 1
root = "/path/to/directory"
studyRoot = os.path.join(root, 'Studies')
os.makedirs(os.path.join(root, 'Normalized'))
studyList = os.listdir(studyRoot)

for study in tqdm(studyList, desc="Study Progress"):
    if study.startswith("."):
        continue

    imageList = os.listdir(os.path.join(studyRoot, study))
    os.makedirs(os.path.join(root, 'Normalized', study), exist_ok = True)

    for image in imageList:
        if os.path.exists(os.path.join(root, 'Normalized', study, image)):
            continue

        elif image.endswith(".nrrd"):
            continue

        try:
            if image.endswith("_Normalized.nii.gz"):
                continue
            
            else:
                print(f"Working on {image}")
                img = nib.load(os.path.join(studyRoot, study, image))
                data = img.get_fdata()
                img1 = img
                data1 = img1.get_fdata()

                for i in range(0, img.shape[0]):
                    for j in range(0, img.shape[1]):
                        for k in range(0, img.shape[2]):
                            if data[i, j, k] == 255:
                                data1[i, j, k] = 1

                            dir1 = os.path.join(root,
                                                'Normalized',
                                                study,
                                                image)

                new_nifti = nib.Nifti1Image(data1.astype(float), img.affine)
                nib.save(new_nifti, dir1)

        except Exception as e:
            print(f"Error f{e} with file {os.path.join(study, image)}")

# Script designed to find scans and segmentations, then remove the slices from
# both where there is no segmentation presence. This is to help improve 
# model training, and can be replicated by having a user manually select
# upper and lower bounds on a scan

rootDir = "/path/to/directory"
studyList = os.listdir(rootDir)

for study in tqdm(studyList, desc="Study Progress"):
    studyPath = os.path.join(rootDir, study)
    fileList = os.listdir(studyPath)

    for file in fileList:
        if file[:-7] + "AN.nii.gz" in fileList:
            maskPath = os.path.join(rootDir, study, file[:-7] + "AN.nii.gz")
            scanPath = os.path.join(rootDir, study, file)
            scanDestPath = os.path.join(rootDir,
                                          study,
                                          file[:-7] + "Trimmed.nii.gz")
            maskDestPath = os.path.join(rootDir,
                                          study,
                                          file[:-7] + "AN_Trimmed.nii.gz")

            mask = nib.load(maskPath)
            maskData = mask.get_fdata()
            maskMask = maskData == 1

            scan = nib.load(scanPath)
            scanData = scan.get_fdata()

            newScan = []
            newMask = []

            for z in tqdm(range(0, scan.shape[2]), desc="Slice Progress"):
                APPEND = False
                for x in range(0, scan.shape[0]):
                    for y in range(0, scan.shape[1]):
                        if maskMask[x, y, z]:
                            APPEND = True
                            newScan.append(scanData[:, :, z])
                            newMask.append(maskData[:, :, z])

                        if APPEND:
                            break

            newNPScan = np.array(newScan)
            newNPMask = np.array(newMask)

            newNPScan = np.moveaxis(newNPScan, 0, 2)
            newNPMask = np.moveaxis(newNPMask, 0, 2)

            newScan = nib.Nifti1Image(newNPScan.astype(float), scan.affine)
            nib.save(newScan, scanDestPath)

            newMask = nib.Nifti1Image(newNPMask.astype(float), mask.affine)
            nib.save(newMask, maskDestPath)
