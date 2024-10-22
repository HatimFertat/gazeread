# gazeread
reading assistant using gaze tracking data and LLM summarization

original github: https://github.com/pperle/gaze-tracking-pipeline

# How to calibrate the webcam:
1) Print a checkboard image (https://pixabay.com/vectors/pattern-checkered-checkerboard-26399/)
2) $ python camera_calibration.py
3) $ Save different images of the checkerboard by pressing 'S'
4) Press ESC key to generate the file 'calibration_matrix.yaml'

You SHOULD generate your own subject calibration images for better performance, here is how:
1) $ cd gaze-data-collection
2) $ python main.py
3) Look at the letter E and press WASD keys depending on the orientation of the letter: UP = W, DOWN = S, LEFT = A, RIGHT = D. (for example if you see 'E' it means RIGHT, if you see '3' it means LEFT, 'ω' means UP, etc.)
NOTE: you can modify the amount of calibration points by changing the numbers in this line: 'rows, cols = 8, 5'
4) after you're done, make check in the 'data/p00/' folder if you have all the images and if they're referenced in the data.csv without duplicate rows
5) go back: $ cd ..

# How to check the gaze tracking on a white screen:
$ python main.py --calibrate_screen_point --method ridge
(sometimes '--method affine' is better?)

# How to check the gaze tracking with pdf and time spent on each line:
$ python test.py
Press 'S' after the camera turns on to start tracking
Press 'P' to pause (you can check the tab in the top left corner under 'File' there is 'Reading Times')
press ESC to quit the window (generates the file 'final_reading_times.txt' that shows the time spent on every line


# Problems and solutions:
I recommend to first try gaze tracking in a white screen to understand.
The gaze tracking is okay, not perfect but okay. More calibration points make it better.
1) Correct the coordinates: maybe the 'convert_screen_to_pdf_coordinates' function isn't good. Because when i look at the first line, it says 'line 9: 1.5s'
(Maybe the draw_grid function could help understand)

2)The gaze tracking isn't precise enough to do line by line tracking
Solution: regroup the lines by groups of 4 ? Maybe by paragraphs ?
Here is how to do grouping by paragraphs:
for block in text_dict:
   if block[6] == 0:  # this is a text block
      lines.append(block)
grouping by lines is better but a bit less simple.

3)Nothing related to calling the LLM is done yet.
- It could be automatic by detecting when the reading time is too long (right now threshold_time = 5 per line, but maybe not the best, and needs to be adapted if we choose to group by 4 lines or by paragraph)
- Simpler solution: Press the 'L' key on the keyboard to call the LLM and it automatically receives as much previously read text as possible with a prompt similar to this:
'Here is the previously read text [ALL PREVIOUS LINES], explain [LAST LINE BEFORE 'L' KEY WAS PRESSED]

