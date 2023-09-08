import dateutil.parser as dparser

text = "Time:\t12:14:53\n"

# Remove tabs and newline characters, and then split the string by the colon
time_parts = dparser.parse(text, fuzzy=True)

print(time_parts)
