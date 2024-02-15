import json
import pandas as pd
from PIL import Image
from aiohttp import ClientSession
from io import BytesIO
import asyncio
from Data.data import *
from Data.model import uddoktaModel, marchentModel




# Nagad Split Function
async def process_nagad_item(nagad_item, nbrtuDict, nagad):
    if nagad_item in nbrtuDict:
        n = {nagad_item: nbrtuDict[nagad_item]}
        nagad.update(n)


# Bkash Split Function
async def process_bkash_item(bkash_item, nbrtuDict, bkash):
    if bkash_item in nbrtuDict:
        b = {bkash_item: nbrtuDict[bkash_item]}
        bkash.update(b)


# Rocket Split Function
async def process_rocket_item(rocket_item, nbrtuDict, rocket):
    if rocket_item in nbrtuDict:
        r = {rocket_item: nbrtuDict[rocket_item]}
        rocket.update(r)


# Tap Split Function
async def process_tap_item(tap_item, nbrtuDict, tap):
    if tap_item in nbrtuDict:
        t = {tap_item: nbrtuDict[tap_item]}
        tap.update(t)


# Upay Split Function
async def process_upay_item(upay_item, nbrtuDict, upay):
    if upay_item in nbrtuDict:
        u = {upay_item: nbrtuDict[upay_item]}
        upay.update(u)


async def getImage(img_url, session):
    async with session.get(img_url) as response:
        img_data = await response.read()
        return BytesIO(img_data)
    

async def detection(model,img_content,confidence):
    img = Image.open(img_content)
    # result = model(img)
    result = model(source=img,device=0,conf=confidence)
    detection = {}
    data = json.loads(result[0].tojson())
    if len(data) == 0:
        res = {"AI": "No Detection"}
        detection.update(res)
    else:
        df = pd.DataFrame(data)
        name_counts = df['name'].value_counts().sort_index()
        
        for name, count in name_counts.items():
            res = {name: count}
            detection.update(res)
    return detection


async def combineAllResult(uddoktaData,marchentData):
    all_result = {}
    all_result.update(uddoktaData)
    all_result.update(marchentData)               
    return all_result


async def prepareUddokta(uddoktaData):
    all_uddokta = {}
    for sku in uddoktaSKU:
        if sku in uddoktaData:
            all_uddokta.update({sku:uddoktaData[sku]})
    return all_uddokta


async def prepareMarchent(marchentData):
    all_marchent = {}
    for sku in marchentSKU:
        if sku in marchentData:
            all_marchent.update({sku:marchentData[sku]})
    return all_marchent


async def prepareResult(uddoktaData,marchentData):
    uddokta = await prepareUddokta(uddoktaData)
    marchent = await prepareMarchent(marchentData)
    allResult = await combineAllResult(uddokta,marchent)
    return allResult

async def mainDet(url):
    async with ClientSession() as session:
        image = await asyncio.create_task(getImage(url, session))
        Tasks = [
                    asyncio.create_task(detection(uddoktaModel, image,0.7)),
                    asyncio.create_task(detection(marchentModel, image,0.8))
                ]
        uddokta,marchent = await asyncio.gather(*Tasks)
        nbrtuDict = await prepareResult(uddokta,marchent)
        for val_item in NBRTU_val:
            if val_item in nbrtuDict:
                nbrtu_validation_single = {val_item: "yes"}
                nbrtuDict.update(nbrtu_validation_single)
        # Remove Extra Items : 
        for nagad_remove_item in ndel_items:
            if nagad_remove_item in nbrtuDict:
                del nbrtuDict[nagad_remove_item]
        nagad = {}
        bkash = {}
        rocket = {}
        tap = {}
        upay = {}
        # Using asyncio.gather to await multiple process functions concurrently
        process_nagad_tasks = [process_nagad_item(nagad_item, nbrtuDict, nagad) for nagad_item in nagad_items]
        process_bkash_tasks = [process_bkash_item(bkash_item, nbrtuDict, bkash) for bkash_item in bkash_items]
        process_rocket_tasks = [process_rocket_item(rocket_item, nbrtuDict, rocket) for rocket_item in rocket_items]
        process_tap_tasks = [process_tap_item(tap_item, nbrtuDict, tap) for tap_item in tap_items]
        process_upay_tasks = [process_upay_item(upay_item, nbrtuDict, upay) for upay_item in upay_items]


        await asyncio.gather(*process_nagad_tasks, *process_bkash_tasks, *process_rocket_tasks, *process_tap_tasks, *process_upay_tasks)


        nagad_detection = {
            'nagad': nagad,
            'bkash': bkash,
            'rocket': rocket,
            'tap': tap,
            'upay': upay
        }

        nagad_result = json.dumps(nagad_detection)

        
        return nagad_result