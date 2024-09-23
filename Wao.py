import subprocess   #allows for me to access the system and use commands 

input_file = r"C:\Users\aharr\OneDrive\Desktop\325 Project\Prompts.txt"  #path to my prompts.txt aka the input for the questions to phi3
output_file = r"C:\Users\aharr\OneDrive\Desktop\325 Project\output.txt"  #path to my output.txt aka the output from the phi to a file
 
command = f'type "{input_file}" | ollama run phi3 > "{output_file}"' #assigning command to run phi3 and store the output inside of the created output.txt file

subprocess.run(command, shell = True)   #executing the command given by me 

print(f"Output is written to the {output_file}")    #printing out that the output was working properly and there were no issues, then can open the output.txt file to check that there was a response


