import json
near = [
("How can I fall asleep faster?","Keep a consistent bedtime, avoid screens for an hour before bed, and keep the room cool and dark.","Go to bed at the same time nightly, skip caffeine after noon, and keep your bedroom dark and quiet."),
("Give one tip for writing a clear email.","Put the main request in the first sentence so the reader knows what you need right away.","State what you need in the subject line and opening line so the reader sees it immediately."),
("Why is the sky blue?","Air molecules scatter short blue wavelengths of sunlight more than red ones, so blue light reaches us from everywhere.","Sunlight scatters off air molecules, and shorter blue wavelengths scatter far more, filling the sky with blue."),
("What is a good way to learn a new language?","Practice a little every day, speak with native speakers early, and learn common phrases before grammar.","Study daily in short sessions, talk with fluent speakers soon, and focus on everyday phrases first."),
("How do I keep basil fresh?","Trim the stems, keep it in a glass of water on the counter, and loosely cover it with a bag.","Cut the stem ends, stand it in water at room temperature, and cover the leaves loosely with plastic."),
("What is recursion in programming?","Recursion is when a function calls itself on a smaller version of the problem until it reaches a base case.","Recursion means a function solves a problem by calling itself on simpler inputs until a stopping case."),
("Suggest a name for a small coffee shop.","How about 'The Daily Grind'? It is friendly, memorable, and hints at coffee.","How about 'Bean There'? It is playful, short, and clearly about coffee."),
("How can I reduce plastic use at home?","Use cloth bags, buy in bulk, and switch to a refillable water bottle.","Carry reusable bags, buy less packaging, and use a metal water bottle."),
("Explain what inflation is in one sentence.","Inflation is the general rise in prices over time, which means money buys less than before.","Inflation is when prices across the economy go up, so each unit of money buys less."),
("What should I pack for a one-day hike?","Bring water, snacks, sunscreen, a light jacket, and a charged phone with a map.","Pack plenty of water, some food, a rain layer, sun protection, and a phone or map."),
("How do I make a cup of tea properly?","Boil fresh water, pour it over the tea, steep for three to five minutes, then remove the leaves.","Heat fresh water to boiling, steep the tea for about four minutes, then take the bag out."),
("Why do we have seasons?","Earth's axis is tilted, so each hemisphere gets more direct sunlight during part of the year.","Because Earth is tilted, the angle of sunlight on each hemisphere changes through the year."),
("Give a tip for public speaking.","Practice out loud several times and slow down; pauses make you sound more confident.","Rehearse aloud beforehand and speak slowly; short pauses help you seem calm and sure."),
("What is a variable in programming?","A variable is a named storage location that holds a value your program can read and change.","A variable is a name that refers to a stored value, which the program can use and update."),
("How can I stay focused while studying?","Work in 25-minute blocks with short breaks, and put your phone in another room.","Study in short timed sessions with breaks, and keep your phone out of reach."),
("What does a router do?","A router forwards data between networks, such as between your home devices and the internet.","A router directs traffic between networks, connecting the devices in your home to the internet."),
("Recommend a way to start exercising.","Start with a 20-minute brisk walk most days, then add strength exercises once it feels easy.","Begin with short daily walks of about 20 minutes, and add bodyweight exercises after a few weeks."),
("What is photosynthesis?","Photosynthesis is how plants use sunlight, water, and carbon dioxide to make sugar and release oxygen.","In photosynthesis, plants turn light, water, and carbon dioxide into glucose, giving off oxygen."),
("How do I remove a coffee stain from a shirt?","Rinse it with cold water from the back, apply a little dish soap, and wash as usual.","Flush the stain with cold water, rub in some liquid detergent, then launder normally."),
("What is the difference between weather and climate?","Weather is the short-term state of the atmosphere; climate is the average pattern over many years.","Weather describes conditions today or this week, while climate describes long-term average conditions."),
("Give a reason to back up files regularly.","Backups protect your work if your device fails, is lost, or gets hit by ransomware.","Regular backups mean you can recover your files after a crash, theft, or malware attack."),
("How do I politely decline a meeting?","Thank them for the invite, say you cannot attend, and offer to read the notes or suggest another time.","Thank the organizer, explain you are unavailable, and propose an alternate time or ask for a summary."),
("What is an API?","An API is a defined set of rules that lets one program request data or actions from another.","An API is an interface that lets software systems talk to each other through agreed requests."),
("How can I save money on groceries?","Plan meals for the whole week, shop with a list, and buy store brands for staples.","Make a weekly meal plan, stick to a shopping list, and choose generic brands for basics."),
("Why do leaves change color in autumn?","Trees stop making green chlorophyll, revealing yellow and orange pigments already in the leaves.","As chlorophyll breaks down in autumn, the hidden yellow and orange pigments in leaves show through."),
("What is a good habit for healthy teeth?","Brush twice a day with fluoride toothpaste and floss once daily.","Brush morning and night with fluoride toothpaste, and floss every day."),
("Explain what a database index is.","An index is a data structure that lets a database find rows quickly without scanning the whole table.","An index helps a database locate matching rows fast instead of reading every row in the table."),
("How should I store bread to keep it fresh?","Keep it in a bread box or paper bag at room temperature, and freeze any you will not eat soon.","Store it at room temperature in a cloth or paper bag, and freeze slices you will not use within days."),
("What is the purpose of a resume?","A resume summarizes your experience and skills so an employer can quickly judge your fit for a role.","A resume gives employers a short overview of your skills and work history to judge if you fit."),
("Give one tip for taking better photos with a phone.","Use natural light from a window and tap the screen to focus on your subject.","Shoot near a window for soft light, and tap your subject on screen to set focus."),
]
ident = [
("What is the tallest animal?","The giraffe is the tallest animal."),
("How many hours are in a day?","There are 24 hours in a day."),
("What color do you get by mixing blue and yellow?","Mixing blue and yellow gives green."),
("What is 7 plus 8?","7 plus 8 is 15."),
("What is the main language spoken in Brazil?","The main language in Brazil is Portuguese."),
("What does a thermometer measure?","A thermometer measures temperature."),
("Give a tip for remembering names.","Repeat the person's name back when you meet them, then use it once in conversation."),
("What is a noun?","A noun is a word that names a person, place, thing, or idea."),
("How do I boil an egg?","Place the egg in boiling water for nine minutes, then cool it in cold water."),
("Why should you wear a seatbelt?","A seatbelt keeps you in place during a crash and greatly reduces serious injury."),
]
rows=[]
for i,(q,a,b) in enumerate(near):
    rows.append(dict(id=f"nt-{i+1:03d}",set="F",stratum="near_tie",question=q,answer1=a,answer2=b,model1="fresh-handwritten",model2="fresh-handwritten",human_winner="tie"))
for i,(q,a) in enumerate(ident):
    rows.append(dict(id=f"id-{i+1:03d}",set="F",stratum="identical",question=q,answer1=a,answer2=a,model1="fresh-handwritten",model2="fresh-handwritten",human_winner="tie"))
r=[max(len(x['answer1']),len(x['answer2']))/min(len(x['answer1']),len(x['answer2'])) for x in rows if x['stratum']=='near_tie']
assert len(rows)==40 and max(r)<=1.10, max(r)
open("set_f.jsonl","w").write("".join(json.dumps(x)+"\n" for x in rows))
print(len(rows), round(max(r),3))
