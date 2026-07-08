# padic-logic.symmachus.org

I would like to have a cute page on a website -- let's call it padic-logic.symmachus.org .

You enter a CSP problem. It reduces it down to the ternary form (A v B  v C) ^ (B ^ C ^ ~D). Then it turns that into a p-adic linear regression problem use the methods of this paper. It calculates how many brute force hyperplanes it will take to find the solution. If the user says to go ahead with it, it will use worker threads in their browser to find a solution. (It will update a few times per second to show its progress.)

What would be really fun: if the user is on Chrome or Edge (where languageModel is available), then you can also take a natural language description of a problem, and it will generate the CSP form (and the proceed with the rest of the task).

Make some images of what this will look like on screen, I'll approve them and then we can try it out. Make a subdirectory for this build, leave this as the README in the subdirectory, and we'll make further plans from there.
