import os
import httpx
import base64
import pandas as pd
import PIL.Image
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from ast import literal_eval

import google.generativeai as genai

# list files in /Task1/train
# print(os.getcwd())


genai.configure(api_key='')

def upload_to_gemini(path, mime_type=None):
  file = genai.upload_file(path, mime_type=mime_type)
  print(f"Uploaded file '{file.display_name}' as: {file.uri}")
  return file

# Create the model
generation_config = {
  "temperature": 0.7, #0.7
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}


def read_images(compounds):
    files=[]
    for compound in compounds:
        for image in os.listdir('Task1/train/'+compound):
            files.append(upload_to_gemini('Task1/train/'+compound+'/'+image, mime_type="image/png"))
    return files

files_idiomatic=read_images(["bad apple", "elbow grease", "night owl"])

history_idiomatic_MAIN=[
    {
      "role": "user",
      "parts": [
        "Given this compound:\nbad apple\nand this sentence:\nThe problem for them, of course, is how to explain how these few bad apples managed to stay in place for so many years.\n\nIs the compound literal or idiomatic? Give me the reasoning and the label\n\n",
      ],
    },
    {
      "role": "model",
      "parts": [
        "The compound \"bad apple\" in the sentence is **idiomatic**.\n\n**Reasoning:**\n\nA literal interpretation of \"bad apple\" would refer to a single piece of fruit that has spoiled. However, in the sentence, \"bad apples\" is used figuratively to represent a small number of corrupt or problematic individuals within a larger group (presumably an organization or team). The meaning isn't derived from the individual words' literal definitions but from a conventional, established figurative usage. The phrase's meaning relies on cultural understanding and shared knowledge of the metaphor.\n\nTherefore, because its meaning is not the sum of its parts but a figurative expression, it's classified as an idiom.\n\n Therefore, it's an idiomatic compound.\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        files_idiomatic[0],
        files_idiomatic[1],
        files_idiomatic[2],
        files_idiomatic[3],
        files_idiomatic[4],
        "For each of these images, extract the most relevant elements and their associated meaning to the compound.\nUse this format:\nImage 1:\n- Relevant Elements\n- Meaning",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Here's a breakdown of the images, focusing on relevant elements and their meanings in relation to the compound \"bad apple\":\n\n**Image 1:**\n\n* **Relevant Elements:** Three children spray-painting graffiti on a wall.  The act is rebellious and possibly illegal.\n* **Meaning:** This image doesn't directly depict \"bad apples,\" but it shows individuals engaging in behavior that could be considered disruptive or problematic within a larger societal context.  It suggests a group of individuals acting against norms.  The connection is loose; it hints at the idea of a small group causing trouble, but doesn't explicitly show \"bad apples\" within a larger, otherwise good group.\n\n**Image 2:**\n\n* **Relevant Elements:** A child accidentally knocks over two cups of coffee and cookies, making a mess.\n* **Meaning:**  This is not directly related to the \"bad apple\" idiom. It shows clumsiness or an accident, not intentional maliciousness or corruption associated with the idiom.\n\n**Image 3:**\n\n* **Relevant Elements:** A perfectly preserved peach, cut in half, showing its juicy, ripe interior.  There's a light dusting of what looks like frost or sugar on the surface.\n* **Meaning:** This image represents the opposite of a \"bad apple.\" It depicts something fresh, wholesome, and good.  It's a visual counterpoint to the idea of decay or corruption.\n\n**Image 4:**\n\n* **Relevant Elements:** A severely rotten and decaying apple, broken into pieces.\n* **Meaning:** This is a direct visual representation of a \"bad apple.\" The decay and disintegration clearly symbolize spoilage and corruption.  It's a strong, literal depiction of the idiom's meaning.\n\n**Image 5:**\n\n* **Relevant Elements:** A burlap sack filled with many seemingly perfect, ripe red apples. A few more apples are placed near the sack.\n* **Meaning:** This image represents a group of mostly good items. The implication is that even within a seemingly good group, there could potentially be a \"bad apple\" hidden (though not explicitly shown in the picture).  It sets the stage for the possibility of a corrupted element among many good ones, aligning with the context of the idiom.\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Based exclusively on the Relevant Elements, order ALL five images from the most representative to the idiomatic meaning of the compound to the least one. Place the image as the last one that is unrelated to the compound.\nLiteral representations of the compound must be placed as the last ones, even though they could have some double meaning or representation since they represent the literal meaning of the compound and not the idiomatic one.\nPlease don't use metaphors to find other meanings. \nPlease don't do any implication.\nPlease don't look for any potential behind what you see. \nYou can base your decision exclusively on the Relevant Elements listed above.\nDo not look for meanings behind the images or for any concepts. If you see a cat, it is a cat and not a representation of the human spirit.",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Based solely on the explicitly described \"Relevant Elements\" and without interpretation or inference, the images ranked from most to least representative of the idiom \"bad apple\" are:\n\n1. **Image 1:** The children's actions depict a group exhibiting disruptive behavior, showing a the idiomatic meaning of \"bad apple\". This strongly relates to the *concept* of \"bad apple\" with a group comming a very bad behaviour\n\n2. **Image 2:** This image is linked to the concept of \"bad apple\" since the boy represented in the figure is making a mess. Compared to the previous image, here the concept of \"bad apple\" is weark\n\n3,**Image 5:** This image shows a collection of apples, matching part of the litteral meaning of the compound but without the negative element. For this reason the image is placed as third one\n\n4. **Image 4:** This is a literal representation of a bad apple. While it's highly relevant to the literal meaning, it's ranked lower because the prompt prioritized the idiomatic meaning first and literal representations last.\n\n3. **Image 3:** This image is completely unrelated to the compound of \"bad apple\". For this reason it is placed as last one.\n",
      ],
    },
	{
      "role": "user",
      "parts": [
        "Given the previous reasoning that you have done, return me your answer as a list of images. The list must include exclusively all five images'name. Example: [xxx.png, xxx.png, xxx.png, xxx.png, xxx.png].\n Return ONLY the list."
      ],
    },
	{
      "role": "model",
      "parts": [
        "[Image1, Image2, Image5, Image4, Image3]"
      ],
    },
	{
      "role": "user",
      "parts": [
        "Given this compound:\nelbow grease\nand this sentence:\nIt took a lot of elbow grease to get the old engine running again.\n\nIs the compound literal or idiomatic? Give me the reasoning and the label\n\n",
      ],
    },
    {
      "role": "model",
      "parts": [
        "The compound \"elbow grease\" in the sentence is **idiomatic**.\n\n**Reasoning:**\n\n* **Literal meaning:**  The literal meaning of \"elbow grease\" would refer to the grease or oil on one's elbows.  This is nonsensical in the context of the sentence.\n\n* **Figurative meaning:** The phrase is used figuratively to mean hard work, strenuous effort, or diligent application of physical labor. This is the only meaningful interpretation within the sentence.\n\nBecause the meaning is not derived from the individual words' literal meanings but instead from a conventional, figurative understanding, \"elbow grease\" functions as an idiom.\n\n Therefore, it's an idiomatic compound.\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        files_idiomatic[5],
        files_idiomatic[6],
        files_idiomatic[7],
        files_idiomatic[8],
        files_idiomatic[9],
        "For each of these images, extract the most relevant elements and their associated meaning to the compound.\nUse this format:\nImage 1:\n- Relevant Elements\n- Meaning",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Here's a breakdown of the images, extracting relevant elements and their meaning in relation to the compound \"elbow grease\":\n\n**Image 1:**\n\n* **Relevant Elements:** A hand scrubbing a heavily soiled stovetop with a sponge.  Visible grime and burnt-on food.\n* **Meaning:** This directly depicts the physical effort and scrubbing required to clean something thoroughly.  It visually represents the \"elbow grease\" aspect of hard work and manual labor.\n\n\n**Image 2:**\n\n* **Relevant Elements:** A gloved hand pouring a viscous, orange liquid (resembling oil or grease) from a rusty pipe into a container.\n* **Meaning:** While not directly related to cleaning, the image suggests manual labor and the handling of potentially messy materials.  The rusty pipe hints at the need for effort in handling and maintenance.  The orange liquid could be metaphorically linked to the \"grease\" part of the idiom, though the connection is weaker than Image 1.\n\n\n**Image 3:**\n\n* **Relevant Elements:** A hand using a brush to clean a gutter, with debris flying off.\n* **Meaning:**  Similar to Image 1, this shows physical effort and cleaning.  The act of brushing away dirt and grime represents the manual labor implied by \"elbow grease.\"\n\n\n**Image 4:**\n\n* **Relevant Elements:** A person wearing protective leg pads.\n* **Meaning:** This shows preparation for physical work, implying the potential for effort and exertion, but does not directly depict the act of cleaning or manual labor itself. The connection to \"elbow grease\" is weak and indirect; it is more about preparation for the activity.\n\n\n**Image 5:**\n\n* **Relevant Elements:** A person applying a viscous, orange liquid (resembling wax or honey) to their arm.\n* **Meaning:**  This depicts the application of a substance, but it's not cleaning or related to the idea of strenuous effort. The connection to \"elbow grease\" is minimal and primarily rests on the visual similarity of the liquid's color to the word \"grease,\" which is far-fetched.\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Based exclusively on the Relevant Elements, order ALL five images from the most representative to the idiomatic meaning of the compound to the least one. Place the image as the last one that is unrelated to the compound.\nLiteral representations of the compound must be placed as the last ones, even though they could have some double meaning or representation since they represent the literal meaning of the compound and not the idiomatic one.\nPlease don't use metaphors to find other meanings. \nPlease don't do any implication.\nPlease don't look for any potential behind what you see. \nYou can base your decision exclusively on the Relevant Elements listed above.\nDo not look for meanings behind the images or for any concepts. If you see a cat, it is a cat and not a representation of the human spirit.",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Based solely on the explicitly described \"Relevant Elements\" and without interpretation or inference, the images ranked from most to least representative of the idiom \"elbow grease\" are:\n\n1. **Image 1:** The direct depiction of scrubbing ingrained dirt is the most literal representation of hard physical work. For this reason this image is the strongly linked to the idiomic concept of \"elbow grease\".\n\n2. **Image 3:** Cleaning a gutter, while less directly related to the idiomic concept of \"elbow grease\", still shows manual labor and effort to remove grime.\n\n3. **Image 2:** This image show handling a greasy substance in a work setting. Although it suggests involvement with the work, the presence of the oily substance associates the image more with the literal meaning.\n\n4. **Image 5:** Applying a liquid to skin is unrelated to the physical effort implied by \"elbow grease\".  It has no relevant elements relating to labor, scrubbing, or grime removal. For this reason the image is place as fourth one.\n\n5**Image 4:** This image is completely unrelated to the compound of \"elbow greas\". For this reason it is placed as last one.\n",
      ],
    },
	{
      "role": "user",
      "parts": [
        "Given the previous reasoning that you have done, return me your answer as a list of images. The list must include exclusively all five images'name. Example: [xxx.png, xxx.png, xxx.png, xxx.png, xxx.png].\n Return ONLY the list."
      ],
    },
	{
      "role": "model",
      "parts": [
        "[Image1, Image3, Image2, Image5, Image4]"
      ],
    },
	{
      "role": "user",
      "parts": [
        "Given this compound:\nnight owl\nand this sentence:\nIt's a constant battle for us, as he is a morning person and I am a night owl, so I find that going to sleep at 9.30 really cuts out the best hours.\n\nIs the compound literal or idiomatic? Give me the reasoning and the label\n\n",
      ],
    },
    {
      "role": "model",
      "parts": [
        "The compound \"night owl\" in the sentence is **idiomatic**.\n\n**Reasoning:**\n\nA literal interpretation of \"night owl\" would be an owl that is active at night (which is true of real owls). However, in the given sentence,  \"night owl\" doesn't refer to a bird. It's used figuratively to describe a person who is most active and alert at night. This figurative meaning is not predictable from the individual words and is established through conventional usage.  Therefore, it's an idiom – a phrase whose meaning isn't obvious from the individual words themselves.\n\n Therefore, it's an idiomatic compound.\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        files_idiomatic[10],
        files_idiomatic[11],
        files_idiomatic[12],
        files_idiomatic[13],
        files_idiomatic[14],
        "For each of these images, extract the most relevant elements and their associated meaning to the compound.\nUse this format:\nImage 1:\n- Relevant Elements\n- Meaning",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Here's a breakdown of the relevant elements and their meanings in relation to the compound \"night owl,\" for each image:\n\n**Image 1:**\n\n* **Relevant Elements:** A large, orange-hued moon, a dark owl with bright orange eyes perched on a bare branch in the night.\n* **Meaning:** This image directly depicts the literal components of \"night owl\"—an owl (the \"owl\" part) at night (the \"night\" part). The moon emphasizes the nighttime setting.  It visually represents the core idea behind the idiom, although it doesn't convey the human behavioral aspect.\n\n\n**Image 2:**\n\n* **Relevant Elements:** A boy working at his computer late at night (implied by the dark background with stars).\n* **Meaning:**  This image directly portrays a human who is awake and active during the nighttime hours, embodying the human behavior associated with the idiom \"night owl.\" The dark background reinforces the nighttime aspect.\n\n\n**Image 3:**\n\n* **Relevant Elements:** An owl sitting on a branch during the daytime, near an autumnal scene.\n* **Meaning:** This image depicts an owl, but not at night. While it contains an \"owl,\" the scene lacks the nighttime component crucial to the meaning of \"night owl.\" Therefore, the image has little to do with the meaning of the idiom.\n\n\n**Image 4:**\n\n* **Relevant Elements:** A child asleep in pajamas on a chair/sofa.\n* **Meaning:** This image represents sleep, which is the opposite of being a \"night owl\". The image does not relate to the compound in meaning.\n\n\n**Image 5:**\n\n* **Relevant Elements:** A dumbbell.\n* **Meaning:**  This image is completely unrelated to the concept of \"night owl.\"  There is no connection between weightlifting and the nocturnal behavior the term describes.\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Based exclusively on the Relevant Elements, order ALL five images from the most representative to the idiomatic meaning of the compound to the least one. Place the image as the last one that is unrelated to the compound.\nLiteral representations of the compound must be placed as the last ones, even though they could have some double meaning or representation since they represent the literal meaning of the compound and not the idiomatic one.\nPlease don't use metaphors to find other meanings. \nPlease don't do any implication.\nPlease don't look for any potential behind what you see. \nYou can base your decision exclusively on the Relevant Elements listed above.\nDo not look for meanings behind the images or for any concepts. If you see a cat, it is a cat and not a representation of the human spirit.",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Based solely on the explicitly stated \"Relevant Elements\" and without interpretation or implication, the order from most to least representative of the idiomatic meaning of \"night owl\" is:\n\n1. **Image 2:** This image directly shows a human active at night, which is the core of the idiom's meaning.\n\n2. **Image 4:** The 'image shows a child in pajamas sitting on a sofa. This suggests that he is awake despite it being bedtime. For this reason, the image can be associated with the idiomatic meaning of compound. At the same time, the concept is less strong than the depiction of the image placed in the first position\n\n3. **Image 3:** Contains an owl, but not at night.  As consequence it partially represent the literal meaning, placing the image in third position.\n\n4. **Image 1:** This image shows a night owl (bird) at night. It's a literal representation of the words, but doesn't show the human behavior described by the idiom.\n\n5.**Image 5:** This image is completely unrelated to the compound of \"night owl\". For this reason it is placed as last one.\n",
      ],
    },
	{
      "role": "user",
      "parts": [
        "Given the previous reasoning that you have done, return me your answer as a list of images. The list must include exclusively all five images'name. Example: [xxx.png, xxx.png, xxx.png, xxx.png, xxx.png].\n Return ONLY the list."
      ],
    },
	{
      "role": "model",
      "parts": [
        "[Image2, Image4, Image3, Image1, Image5]"
      ],
    },
]


files_literal=read_images(["pain in the neck", "silver bullet", "black box"])
history_literal_MAIN=[
    {
      "role": "user",
      "parts": [
        "Given this compound:\ntwo-way street\nand this sentence:\nIn many cases, two-way streets result in twice as many pedestrian accidents as one way.\nIs the compound literal or idiomatic? Give me the reasoning and the label ",
      ],
    },
    {
      "role": "model",
      "parts": [
        "The compound \"two-way street\" in the sentence is **literal**.\n\n**Reasoning:**\n\n* **Literal Meaning:** In this context, \"two-way street\" refers to a physical road that allows traffic to flow in both directions. The sentence is discussing the impact of road design on pedestrian accidents. It's comparing streets where cars can travel in two directions to those where traffic only flows in one.\n* **No Figurative Language:** The compound is not used to imply something else. It does not relate to relationships, reciprocal agreements or something symbolic as it often can in idiomatic contexts. There is no figurative meaning being implied. The meaning of \"two-way street\" in this sentence is exactly how you would describe an actual two-way road.\n\n**Therefore, the compound \"two-way street\" is used in a literal sense in this sentence.**\n\nTherefore, it's a literal compound.\n",
      ],
    },
    {
      "role": "user",
      "parts": [
	    files_literal[0],
        files_literal[1],
        files_literal[2],
        files_literal[3],
        files_literal[4],
        "For each of these images, extract the most relevant elements and their associated meaning to the compound.\nUse this format:\nImage 1:\n- Relevant Elements\n- Meaning",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Okay, here's the analysis of each image, focusing on relevant elements and their meaning related to the compound \"two-way street\":\n\n**Image 1:**\n\n*   **Relevant Elements:**\n    *   A road with a yellow line down the center.\n    *   Vehicles traveling in opposite directions on either side of the line.\n    *   Buildings and trees lining either side of the road.\n*   **Meaning:** This image depicts the literal meaning of a \"two-way street.\" The central line clearly divides the road into two lanes, enabling traffic to move in opposing directions. It's a straightforward visual representation of a road with two-way traffic.\n\n**Image 2:**\n\n*   **Relevant Elements:**\n    *   A compass with a prominent \"North\" (N) and \"South\" (S) marking.\n    *  A dial with markings and needle\n*   **Meaning:** This image doesn't directly relate to the literal meaning of \"two-way street.\" However, it can be *idiomatically* interpreted as relating to a concept of direction, or flow in two directions. A compass represents different paths or direction. In this image, it shows the direction of North and South, therefore, *potentially* implying direction in two directions. This is not commonly associated with the term \"two-way street,\" thus the association would be more abstract.\n\n**Image 3:**\n\n*   **Relevant Elements:**\n    *   Two people, one offering a drink and another accepting it\n    *   A drink being offered and accepted\n*   **Meaning:** This image *idiomatically* represents a give-and-take relationship. The exchange of drinks represents a reciprocal action. This is a common metaphorical usage of the term \"two-way street.\" Where both parties give and receive equally.\n\n**Image 4:**\n\n*   **Relevant Elements:**\n    *   A road with orange cones\n    *   Construction machinery\n    *   Traffic lights\n*   **Meaning:** This image shows a literal road, similar to the first image. This one is under construction, so while traffic is present, it might not be in full operation as a 'two-way street'. However, it still represents the concept of two-way traffic, but now it is associated with disruptions and road work.\n\n**Image 5:**\n\n*   **Relevant Elements:**\n     *   Two people shaking hands.\n     *   Smiling faces\n*   **Meaning:** This image *idiomatically* represents the concept of mutual agreement or cooperation. Shaking hands is a visual way to express mutual agreement and reciprocation. It reinforces the idiomatic use of \"two-way street\" where both parties give and receive.\n\nLet me know if you have any other images to analyze!\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Based exclusively on the Relevant Elements, order ALL five images from the most representative to the literal meaning of the compound to the least one.\nIdiomatic representations of the compound must be placed as the last ones, even though they could have some double meaning or representation since they represent the idiomatic meaning of the compound and not the literal one.\nPlease don't use metaphors to find other meanings. \nPlease don't do any implication.\nPlease don't look for any potential behind what you see. \nYou can base your decision exclusively on the Relevant Elements listed above.\nDo not look for meanings behind the images or for any concepts. If you see a cat, it is a cat and not a representation of the human spirit.",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Okay, based *exclusively* on the \"Relevant Elements\" I previously listed, and following your strict guidelines to avoid interpretation or implied meaning, here's the order from most to least representative of the *literal* meaning of \"two-way street,\" placing idiomatic representations last:\n\n1.  **Image 1:**\n    *   **Relevant Elements:** A road with a yellow line down the center, vehicles traveling in opposite directions, buildings and trees.\n    *   This image *directly* depicts a road with traffic flowing in two opposing directions, which is the core, literal meaning of \"two-way street\".\n\n2.  **Image 4:**\n    *   **Relevant Elements:** A road with orange cones, construction machinery, traffic lights.\n    *   This image shows a road, which relates to the \"street\" part of the compound. Although it depicts a road under construction, the relevant element is that it is a road. It implies the concept of two-way traffic.\n\n3.  **Image 3:**\n    *   **Relevant Elements:** Two people, one offering a drink and another accepting it, a drink being offered and accepted.\n     *   This image represents a reciprocal action, so it is not a street, therefore, it is an idiomatic use of \"two-way street\", it has no literal sense.\n\n4.  **Image 5:**\n    *   **Relevant Elements:** Two people shaking hands, smiling faces.\n    *   This image represents an agreement or cooperation; it has no relationship with the literal elements of the compound \"two-way street\", while it has a strong connection to the idiomic meaning of the compound. Therefore, this is the least representative of the literal meaning of the compound.\n\n5.  **Image 2:**\n    *   **Relevant Elements:** A compass with a prominent \"North\" (N) and \"South\" (S) marking, a dial with markings and a needle.\n    *   This image is completely unrelated to the compound of \"two-way street\". For this reason it is placed as last one.\n",
      ],
    },
	{
      "role": "user",
      "parts": [
        "Given the previous reasoning that you have done, return me your answer as a list of images. The list must include exclusively all five images'name. Example: [xxx.png, xxx.png, xxx.png, xxx.png, xxx.png].\n Return ONLY the list."
      ],
    },
	{
      "role": "model",
      "parts": [
        "[Image1, Image4, Image3, Image5, Image2]"
      ],
    },
	{
      "role": "user",
      "parts": [
        "Given this compound:\nsilver bullet\nand this sentence:\nChastel had loaded a custom made silver bullet into his gun with which he fired on the creature and killed the beast with a single shot.\nIs the compound literal or idiomatic? Give me the reasoning and the label",
      ],
    },
    {
      "role": "model",
      "parts": [
        "The compound \"silver bullet\" in the sentence is **literal**.\n\n**Reasoning:**\n\n* **Physical Object:** The sentence explicitly states that Chastel \"loaded a custom-made silver bullet into his gun.\" This clearly indicates a physical object made of silver that is used as ammunition.\n*   **Direct Function:** The bullet's function is to be fired from a gun and used to kill a creature. This is the standard function of a literal bullet.\n*   **No Figurative Meaning:** There is no indication in the sentence that \"silver bullet\" is meant to symbolize a solution to a problem, a quick fix, or anything else that departs from the literal meaning. It's described in the narrative as an actual projectile made of silver.\n\n**Idiomatic Meaning (for contrast):**\nThe idiomatic use of \"silver bullet\" implies a simple, quick, and almost magical solution to a difficult problem. For example: \"There's no silver bullet to fix the economy.\" In that case, \"silver bullet\" is being used figuratively to mean a single, easy fix, not a literal projectile.\n\n**In Summary**\nThe sentence uses \"silver bullet\" to refer to a specific and literal object; a custom-made bullet made of silver, thus it is being used literally.\n\nTherefore, it's a literal compound.\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        files_literal[5],
        files_literal[6],
        files_literal[7],
        files_literal[8],
        files_literal[9],
        "For each of these images, extract the most relevant elements and their associated meaning to the compound.\nUse this format:\nImage 1:\n- Relevant Elements\n- Meaning",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Okay, here's an analysis of each image, focusing on relevant elements and their meaning in relation to the compound \"silver bullet,\" keeping in mind we determined it to be literal in the previous example:\n\n**Image 1:**\n\n-   **Relevant Elements:** A syringe filled with clear liquid, a vial labeled \"Almightie Cure\", and another small syringe, both partially orange in color, are visible. The syringe has a needle and graduations on its side.\n-   **Meaning:** The syringe and vial represent medical treatment and the literal injection of a substance. This is in contrast to a physical bullet. While it is something that can be injected into a body, it is not a projectile fired from a gun.\n\n**Image 2:**\n\n-   **Relevant Elements:** A dark, sleek object resembling a pen or a bullet. The bottom is pointed and the top is capped. It is made of metal and has visible threads.\n-   **Meaning:** This image strongly suggests a literal bullet, given the overall shape, pointed tip, and appearance. The metal and general design are evocative of actual ammunition. It is not silver, but shares a similar shape.\n\n**Image 3:**\n\n-   **Relevant Elements:** A roll of clear tape. It has an orange piece at the edge.\n-   **Meaning:** The tape is a tool for adhering things together and has no relation to a bullet. It's not used for penetration or causing harm, thus has no relation to the idea of a literal bullet.\n\n**Image 4:**\n\n-   **Relevant Elements:** A belt containing multiple rounds of ammunition. The casings are red and the belt is made of leather and metal.\n-   **Meaning:** The image shows a set of literal bullets or shotgun shells in a carrier. These are all designed to be projectiles that can be fired from a firearm.\n\n**Image 5:**\n\n-   **Relevant Elements:** A man writing with a pen or pencil. He is sitting at a desk.\n-   **Meaning:** The man is using a writing implement and does not appear to be using or in possession of a bullet. The image is unrelated to a bullet, literal or otherwise.\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Based exclusively on the Relevant Elements, order ALL five images from the most representative to the literal meaning of the compound to the least one.\nIdiomatic representations of the compound must be placed as the last ones, even though they could have some double meaning or representation since they represent the idiomatic meaning of the compound and not the literal one.\nPlease don't use metaphors to find other meanings. \nPlease don't do any implication.\nPlease don't look for any potential behind what you see. \nYou can base your decision exclusively on the Relevant Elements listed above.\nDo not look for meanings behind the images or for any concepts. If you see a cat, it is a cat and not a representation of the human spirit.",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Okay, based *exclusively* on the \"Relevant Elements\" listed in the previous analysis and without any inference or interpretation beyond the literal, here is the order from most representative of a literal \"silver bullet\" to least, placing idiomatic representations last:\n\n1.  **Image 2:** (Dark, sleek object resembling a pen or bullet). The relevant elements are a pointed tip, and a shape consistent with a bullet. It is missing the silver color, but the form and texture suggest a projectile.\n2. **Image 4:** (Ammunition belt with multiple rounds). The relevant elements are literal rounds of ammunition, which directly correspond to the literal meaning of \"bullet\". It does not explicitly say it is silver, but matches the function of the word.\n3.  **Image 1:** (Syringe and vial). The relevant elements are the syringe with a needle and medical liquid. It is related to injections and bodily fluids and doesn't have anything related to a bullet.\n4.  **Image 5:** (Man writing with a pen). The relevant element is a man using a pen. This image is most removed from the idea of any kind of bullet, having no elements that correspond to the function or concept of a bullet.\n5.  **Image 3:** (Roll of clear tape). The relevant element is the clear tape, a material used for adhesion, and without a direct correlation to a bullet, whether literal or idiomatic. It has no elements of a projectile. This image is completely unrelated to the compound of \"silver bullet\". For this reason it is placed as last one.\n",
      ],
    },
	{
      "role": "user",
      "parts": [
        "Given the previous reasoning that you have done, return me your answer as a list of images. The list must include exclusively all five images'name. Example: [xxx.png, xxx.png, xxx.png, xxx.png, xxx.png].\n Return ONLY the list."
      ],
    },
	{
      "role": "model",
      "parts": [
        "[Image2, Image4, Image2, Image5, Image3]"
      ],
    },
	{
      "role": "user",
      "parts": [
        "Given this compound:\nblack box\nand this sentence:\nThe unit is encased in an attractive anodized black box that provides EMI/RFI shielding and allows for secure mounting with its threaded inserts on the bottom of its chassis.\nIs the compound literal or idiomatic? Give me the reasoning and the label",
      ],
    },
    {
      "role": "model",
      "parts": [
        "The compound \"black box\" in the sentence is **literal**.\n\n**Reasoning:**\n\n***Literal Meaning:** In its literal sense, \"black box\" would simply mean a box that is black in color.\n*   **Contextual Meaning:** The sentence describes the unit as being \"encased in an attractive anodized black box.\" Here, \"black\" is functioning as a descriptive adjective referring to the color of the box, and \"box\" is the literal physical object of a container. Therefore, the compound functions as a noun phrase describing a box that happens to be black.\n\n**Reasoning:**\n\nThe sentence is straightforward and describes the physical properties of the enclosure. There is no suggestion of any hidden meaning or symbolic interpretation. The box is literally black, and its physical function as a container is made explicit.\n\n**Label:**\n\n**Literal**\n\nTherefore, it's a literal compound.\n",
      ],
    },
    {
      "role": "user",
      "parts": [
	    files_literal[10],
        files_literal[11],
        files_literal[12],
        files_literal[13],
        files_literal[14],
        "For each of these images, extract the most relevant elements and their associated meaning to the compound.\nUse this format:\nImage 1:\nRelevant Elements\nMeaning",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Okay, here's the analysis of each image focusing on the elements relevant to the compound \"black box\":\n\nImage 1:\nRelevant Elements:\n*   A box with a dark color.\n*   The box is a closed container.\nMeaning:\nThe box is literally black in color and is depicted as a physical container.\n\nImage 2:\nRelevant Elements:\n* An open cardboard container.\nMeaning:\nThe container is not black, and is a typical cardboard box.\n\nImage 3:\nRelevant Elements:\n* A brown and white dog.\nMeaning:\nThe animal has no relation to the compound \"black box\".\n\nImage 4:\nRelevant Elements:\n* A computer monitor with a black screen\nMeaning:\nThe screen of the monitor is black but is not a box.\n\nImage 5:\nRelevant Elements:\n* A machine with an opening where a sock is coming out\nMeaning:\nThe machine has an opening, but is not a literal box. The opening is black.\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Based exclusively on the Relevant Elements, order ALL five images from the most representative to the literal meaning of the compound to the least one.\nIdiomatic representations of the compound must be placed as the last ones, even though they could have some double meaning or representation since they represent the idiomatic meaning of the compound and not the literal one.\nPlease don't use metaphors to find other meanings. \nPlease don't do any implication.\nPlease don't look for any potential behind what you see. \nYou can base your decision exclusively on the Relevant Elements listed above.\nDo not look for meanings behind the images or for any concepts. If you see a cat, it is a cat and not a representation of the human spirit. ",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Okay, based *exclusively* on the \"Relevant Elements\" listed in the previous analysis and without any inference or interpretation beyond the literal, here is the order from most representative of a literal \"black box\" to least, placing idiomatic representations last:\n\n1.  **Image 1:** \n    *   Relevant Elements: A box with a dark color. The box is a closed container.\n    *   Reasoning: This is the most literal representation. It shows a container that is black, directly matching the words in the compound.\n\n2.  **Image 2:**\n    *   Relevant Elements: An open cardboard container.\n    *   Reasoning: While it is a box, it is not black.\n\n3.  **Image 4:**\n    *   Relevant Elements: A computer monitor with a black screen.\n    *   Reasoning: While there is black present, the object is not a box, it's a screen.\n\n4.  **Image 5:**\n     *   Relevant Elements: A machine with an opening where a sock is coming out.\n     *   Reasoning: The object is not a box and the opening is the only place where the color black is present.\n\n5. **Image 3:**\n    *   Relevant Elements: A brown and white dog.\n    *   Reasoning: This has no relation to the compound \"black box\" at all. It's not a box, nor is it primarily black. This image is completely unrelated to the compound of \"silver bullet\". For this reason it is placed as last one.\n",
      ],
    },
	{
      "role": "user",
      "parts": [
        "Given the previous reasoning that you have done, return me your answer as a list of images. The list must include exclusively all five images'name. Example: [xxx.png, xxx.png, xxx.png, xxx.png, xxx.png].\n Return ONLY the list."
      ],
    },
	{
      "role": "model",
      "parts": [
        "[Image1, Image2, Image4, Image5, Image3]"
      ],
    },
]

tot=0
correct=0

top1=0

split="xeval"

# MODEL_NAME="gemini-1.5-flash"
MODEL_NAME="gemini-2.0-flash-exp"
# MODEL_NAME="gemini-2.0-flash-thinking-exp-01-21"

df= pd.read_csv(f'Task1/{split}/subtask_a_{split}.tsv', delimiter='\t',encoding='utf-8')
labels_reasonings=pd.read_csv(f'Task1/{split}/labels_reasonings_G2T.csv', header=None)
labels_reasonings.columns=['Compound','Reasoning','Label', 'Gt']
# print(labels_reasonings.head())
predictions=[]
compounds=[]
print("")
if os.path.exists(f'results_CoT_FS/{MODEL_NAME}/submission_EN.tsv') and split=="test":
    computed=pd.read_csv(f'results_CoT_FS/{MODEL_NAME}/submission_EN.tsv', delimiter='\t',encoding='utf-8')
    computed_compounds=computed['compound'].values.tolist()
    computed_predictions=computed['expected_order'].values.tolist()
    print("Computed compounds TEST", computed_compounds)
elif os.path.exists(f'results_CoT_FS/{MODEL_NAME}/submission_XE.tsv') and split=="xeval":
    computed=pd.read_csv(f'results_CoT_FS/{MODEL_NAME}/submission_XE.tsv', delimiter='\t',encoding='utf-8')
    computed_compounds=computed['compound'].values.tolist()
    computed_predictions=computed['expected_order'].values.tolist()
    print("Computed compounds XEVAL", computed_compounds)
else:
    print("No computed compounds")
    computed_compounds=[]
    computed_predictions=[]

idx=1
coompounds_list=df['compound'].values.tolist()




for compound in coompounds_list:

    print("-"*50)
    print(idx, compound)
    idx+=1

    just_images=[]
    for image in os.listdir(f'Task1/{split}/'+compound.replace("'","_")):
        just_images.append(image)

    association={}
    for i,el in enumerate(just_images):
        # association_string+=f"IMAGE{i+1}: {el}\n"
        association[f"Image{i+1}"]=el

    
    if compound == 'cat_s eyes':
        compound = "cat's eyes"
    if compound == 'dog_s dinner':
        compound = "dog's dinner"
    if compound == 'devil_s advocate':
        compound = "devil's advocate"
    if compound == 'pig_s ear':
        compound = "pig's ear"

    
    if compound in computed_compounds:
        print("Already computed")
        continue
    compounds.append(compound)

    try:
        row = labels_reasonings[labels_reasonings['Compound']==compound]
        label = row['Label'].values[0]
        reasoning = row['Reasoning'].values[0]
    except:
        print(compound)
        print(row)

    sentence = df[df['compound']==compound]['sentence'].values[0]
    image1_caption = df[df['compound']==compound]['image1_caption'].values[0]
    image2_caption = df[df['compound']==compound]['image2_caption'].values[0]
    image3_caption = df[df['compound']==compound]['image3_caption'].values[0]
    image4_caption = df[df['compound']==compound]['image4_caption'].values[0]
    image5_caption = df[df['compound']==compound]['image5_caption'].values[0]
    print("P:", label)
    # print(expected_order)
    history_idiomatic=history_idiomatic_MAIN.copy()
    history_literal=history_literal_MAIN.copy()

    if label=='idiomatic':
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=generation_config,
        )
        files2=[]
        for image in os.listdir(f'Task1/{split}/'+compound.replace("'","_")):
            files2.append(upload_to_gemini(f'Task1/{split}/'+compound.replace("'","_")+'/'+image, mime_type="image/png"))
        # print(expected_order)

        history_idiomatic.append(
            {
                "role": "user",
                "parts": [
                    f"Given this compound:\n{compound}\nand this sentence:\n{sentence}\n\nIs the compound literal or idiomatic? Give me the reasoning and the label\n\n",
                ],
            }
        )
        history_idiomatic.append(
            {
                "role": "model",
                "parts": [
                    f"The compound \"{compound}\" in the sentence is **{label}**.\n\n**Reasoning:**\n\n{reasoning}\n\nTherefore, it's an {label} compound.\n",
                ],
            }
        )
        history_idiomatic.append(
            {
                "role": "user",
                "parts": [
                    files2[0],
                    files2[1],
                    files2[2],
                    files2[3],
                    files2[4],
                ],
            }
        )
        chat_session = model.start_chat(history=history_idiomatic)
        q1="For each of these images, extract the most relevant elements and their associated meaning to the compound.\nUse this format:\nImage 1:\n- Relevant Elements\n- Meaning"
        response = chat_session.send_message(q1)
        # print(chat_session.history)
        ans1=response.text
        # print(ans1)

        q2="Based exclusively on the Relevant Elements, order ALL five images from the most representative to the idiomatic meaning of the compound to the least one. Place the image as the last one that is unrelated to the compound.\nLiteral representations of the compound must be placed as the last ones, even though they could have some double meaning or representation since they represent the literal meaning of the compound and not the idiomatic one.\nPlease don't use metaphors to find other meanings. \nPlease don't do any implication.\nPlease don't look for any potential behind what you see. \nYou can base your decision exclusively on the Relevant Elements listed above.\nDo not look for meanings behind the images or for any concepts. If you see a cat, it is a cat and not a representation of the human spirit."
        response = chat_session.send_message(q2)
        # print(chat_session.history)
        ans2=response.text
        # print(ans2)

        q3="Given the previous reasoning that you have done, return me your answer as a list. The list must include all five images. Example: [Image1, Image2, Image3, Image4, Image5].\n Return ONLY the list."
        response = chat_session.send_message(q3)
        # print(chat_session.history)
        ans3=response.text
        # print(ans3)
        ans3_list=[el.replace(" ","").replace("[","").replace("]","").strip() for el in ans3.split(",")]
        
        pred=[]
        for i in ans3_list:
            pred.append(association[i])
        print(pred)
        predictions.append(pred)
        # break
        # if pred[0]==expected_order[0]:
        #     correct+=1
        # tot+=1
    else:
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=generation_config,
        )
        files2=[]
        for image in os.listdir(f'Task1/{split}/'+compound.replace("'","_")):
            files2.append(upload_to_gemini(f'Task1/{split}/'+compound.replace("'","_")+'/'+image, mime_type="image/png"))
        # print(expected_order)

        history_literal.append(
            {
                "role": "user",
                "parts": [
                    f"Given this compound:\n{compound}\nand this sentence:\n{sentence}\n\nIs the compound literal or idiomatic? Give me the reasoning and the label\n\n",
                ],
            }
        )
        history_literal.append(
            {
                "role": "model",
                "parts": [
                    f"The compound \"{compound}\" in the sentence is **{label}**.\n\n**Reasoning:**\n\n{reasoning}\n\nTherefore, it's a {label} compound.\n",
                ],
            }
        )
        history_literal.append(
            {
                "role": "user",
                "parts": [
                    files2[0],
                    files2[1],
                    files2[2],
                    files2[3],
                    files2[4],
                ],
            }
        )
        chat_session = model.start_chat(history=history_literal)
        q1="For each of these images, extract the most relevant elements and their associated meaning to the compound.\nUse this format:\nImage 1:\n- Relevant Elements\n- Meaning"
        response = chat_session.send_message(q1)
        # print(chat_session.history)
        ans1=response.text
        # print(ans1)

        q2="Based exclusively on the Relevant Elements, order ALL five images from the most representative to the literal meaning of the compound to the least one. Idiomatic representations of the compound must be placed as the last ones, even though they could have some double meaning or representation since they represent the idiomatic meaning of the compound and not the literal one. Please don't use metaphors to find other meanings. Please don't do any implication. Please don't look for any potential behind what you see. You can base your decision exclusively on the Relevant Elements listed above. Do not look for meanings behind the images or for any concepts. If you see a cat, it is a cat and not a representation of the human spirit."
        response = chat_session.send_message(q2)
        # print(chat_session.history)
        ans2=response.text
        # print(ans2)

        q3="Given the previous reasoning that you have done, return me your answer as a list. The list must include all five images. Example: [Image1, Image2, Image3, Image4, Image5].\n Return ONLY the list."
        response = chat_session.send_message(q3)
        # print(chat_session.history)
        ans3=response.text
        # print(ans3)
        ans3_list=[el.replace(" ","").replace("[","").replace("]","").strip() for el in ans3.split(",")]
        
        pred=[]
        for i in ans3_list:
            pred.append(association[i])
        print(pred)
        predictions.append(pred)
        # break
        # if pred[0]==expected_order[0]:
        #     correct+=1
        # tot+=1

    # print("Accuracy:", correct/tot)    
    computed_predictions.append(pred)
    computed_compounds.append(compound)
    df_save=pd.DataFrame({'compound':computed_compounds,'expected_order':computed_predictions})
    os.makedirs(f'results_CoT_FS/{MODEL_NAME}', exist_ok=True)
    # os.makedirs(f'results/{MODEL_NAME}/{split}', exist_ok=True)
    if split=="test":
        df_save.to_csv(f'results_CoT_FS/{MODEL_NAME}/submission_EN.tsv', sep='\t',index=False, header=True)
    else:
        df_save.to_csv(f'results_CoT_FS/{MODEL_NAME}/submission_XE.tsv', sep='\t',index=False, header=True)