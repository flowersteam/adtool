from manim import Scene, Text, VGroup, Write, FadeIn, DOWN, BLUE, Square, Arrow, Create, RIGHT, LEFT, UP
from manim import SurroundingRectangle, Rectangle, AnimationGroup, RED, FadeOut, PI, CurvedArrow,YELLOW,Circumscribe

from manim import ImageMobject, Group

class s1(Scene):
    def construct(self):
        # Load the GitHub image
        github_logo = ImageMobject("github.png").scale(0.25)
        
        # Title and Subtitles
        title = Text("flowersteam/adtool").scale(1.5)
        subtitle_line1 = Text("A Python Framework for Assisted Identification", t2c={"Assisted Identification": BLUE}).scale(0.5)
        subtitle_line2 = Text("of Original Settings for Parameterized Systems", t2c={"Original Settings": BLUE}).scale(0.5)
        
        # Positioning
        title_logo_group = Group(github_logo, title).arrange(RIGHT, buff=0.5)
        title_logo_group.move_to(UP*2)
        
        subtitle_line1.next_to(title_logo_group, DOWN)
        subtitle_line2.next_to(subtitle_line1, DOWN)
        
        # Creating a VGroup for the subtitles
        subtitles = VGroup(subtitle_line1, subtitle_line2)

        self.play(FadeIn(github_logo),FadeIn(title), run_time=1)
        self.play(FadeIn(subtitles), run_time=1)  # Writing both subtitles at the same time
        
        self.wait(3)






class s2(Scene):
    def construct(self):
        # Create the introductory text
        intro_text1 = Text("Suppose you want to discover", font_size=20)
        intro_text2 = Text("original parameterizations of a given system", font_size=20)
        intro_text1.to_edge(UP)
        intro_text2.next_to(intro_text1, DOWN)

        # Create the text labels
        alpha_beta = Text("α, β", font_size=36)
        alpha_prime_beta_prime = Text("α', β'", font_size=36)
        alpha_double_prime_beta_double_prime = Text('α", β"', font_size=36)

        # Create the rectangles with the text inside them
        box1 = Rectangle(width=alpha_double_prime_beta_double_prime.width, height=alpha_double_prime_beta_double_prime.height, stroke_opacity=0)
        box2 = Rectangle(width=alpha_double_prime_beta_double_prime.width, height=alpha_double_prime_beta_double_prime.height, stroke_opacity=0)
        box3 = Rectangle(width=alpha_double_prime_beta_double_prime.width, height=alpha_double_prime_beta_double_prime.height, stroke_opacity=0)

        # Group the text with the boxes
        text_box1 = VGroup(box1, alpha_beta)
        text_box2 = VGroup(box2, alpha_prime_beta_prime)
        text_box3 = VGroup(box3, alpha_double_prime_beta_double_prime)

        # Position the text boxes
        text_box1.to_edge(LEFT).shift(UP * 1)
        text_box2.to_edge(LEFT)
        text_box3.to_edge(LEFT).shift(DOWN * 1)

        # Create the rectangles
        rect1 = Square()
        rect2 = Square()
        rect3 = Square()

        # Position the rectangles on the right side of the screen
        rect1.to_edge(RIGHT).shift(UP * 2.5)
        rect2.to_edge(RIGHT)
        rect3.to_edge(RIGHT).shift(DOWN * 2.5)

        # Create arrows
        arrow1 = Arrow(start=text_box1.get_right(), end=rect1.get_left())
        arrow2 = Arrow(start=text_box2.get_right(), end=rect2.get_left())
        arrow3 = Arrow(start=text_box3.get_right(), end=rect3.get_left())

        # Create labels for the groups
        parameters_label = Text("parameters", font_size=24)
        resulting_system_label = Text("resulting system", font_size=24)

        # Position the labels below the groups
        parameters_label.next_to(text_box3, DOWN)
        resulting_system_label.next_to(rect3, DOWN)

        # Add elements to the scene
        self.play(FadeIn(intro_text1))
        self.play(FadeIn(intro_text2))
        self.wait(1)
        self.play(
            AnimationGroup(
                Write(text_box1),
                Write(text_box2),
                Write(text_box3),
                Create(rect1),
                Create(rect2),
                Create(rect3),
                FadeIn(parameters_label), 
                FadeIn(resulting_system_label),
                lag_ratio=0
            )
        )
        self.wait(1)
        self.play(Create(arrow1), Create(arrow2), Create(arrow3))
        self.wait(1)
        # but the sampling might me tricky
        objection_text = Text("but the maping might be non-trivial", font_size=20, 
                                t2c={"non-trivial": RED})
        objection_text.to_edge(DOWN).shift(LEFT * 1)
        self.play(FadeIn(objection_text))
        self.wait(3)


from manim import Scene, Text, VGroup, Write, FadeIn, DOWN, BLUE, Square, Arrow, Create, RIGHT, LEFT, UP
from manim import SurroundingRectangle, Rectangle, AnimationGroup, RED, PI, CurvedArrow

class s3(Scene):
    def construct(self):
        # Check ADTool text
        check_adtool = Text("Check ADTool", font_size=36, t2c={"ADTool": BLUE})
        check_adtool.to_edge(UP)
        self.play(FadeIn(check_adtool))

        # subtext
        subtext = Text("Sampling cycle based on IMGEP algorithm ", font_size=20)
        subtext.next_to(check_adtool, DOWN)
        self.play(FadeIn(subtext))

        # Create boxes for the spaces
        param_space_box = Rectangle(width=3, height=1, stroke_opacity=0)
        obs_space_box = Rectangle(width=3, height=1, stroke_opacity=0)
        goal_space_box = Rectangle(width=3, height=1, stroke_opacity=0) 

        # Position the boxes
        param_space_box.move_to(LEFT * 4).shift(UP * 1)
        obs_space_box.move_to(RIGHT * 4).shift(UP * 1)
        goal_space_box.move_to(DOWN * 3 ).shift(UP * 1)

        # Create labels for the boxes
        param_space_label = Text("Parameters Space", font_size=20)
        obs_space_label = Text("Observation Space", font_size=20)
        goal_space_label = Text("Goal Space", font_size=20)

        # Position the labels
        param_space_label.move_to(param_space_box.get_center())
        obs_space_label.move_to(obs_space_box.get_center())
        goal_space_label.move_to(goal_space_box.get_center())

        # Create arrows
        arrow1 = Arrow(param_space_box.get_right(), obs_space_box.get_left())
        arrow2 = Arrow(obs_space_box.get_bottom(), goal_space_box.get_right())
        arrow3 = Arrow(goal_space_box.get_left(), param_space_box.get_bottom())

        # Add elements to the scene
        # self.play(FadeIn(param_space_box), FadeIn(obs_space_box), FadeIn(goal_space_box))
        self.play(FadeIn(param_space_label), FadeIn(obs_space_label), FadeIn(goal_space_label),
                    FadeIn(arrow1), FadeIn(arrow2), FadeIn(arrow3))



        start_sampling = Text("Start by sampling a random goal", font_size=20)
        start_sampling.move_to(DOWN * 3)

        # make the goal box glow
        self.play( FadeIn(start_sampling),
            Circumscribe(goal_space_box, color=YELLOW, buff=0.1,fade_out=True))
                  


        self.wait(3)
        self.play(FadeOut(start_sampling))
        

        start_sampling = Text("Sample parameters candidates to reach the goal", font_size=20) 
        start_sampling.move_to(DOWN * 3)
        self.play( FadeIn(start_sampling),
            Circumscribe(param_space_box, color=YELLOW, buff=0.1,fade_out=True))
        
        self.wait(3)
        self.play(FadeOut(start_sampling))

        start_sampling = Text("Simulate your system with the sampled parameters", font_size=20)
        start_sampling.move_to(DOWN * 3)

        self.play( FadeIn(start_sampling),
            Circumscribe(obs_space_box, color=YELLOW, buff=0.1,fade_out=True))
        
        self.wait(2)
        self.play(FadeOut(start_sampling))

        start_sampling = Text("Map the observation to the goal space and sample a new goal accordingly", font_size=20)
        start_sampling.move_to(DOWN * 3)

        self.play( FadeIn(start_sampling),
            Circumscribe(goal_space_box, color=YELLOW, buff=0.1,fade_out=True))
        
        self.wait(3)
        self.play(FadeOut(start_sampling))


        # Hold the final scene for a while
        self.wait(3) 



class s4(Scene):
    # accelerated example on the Gray-Scott diffusion system
    def construct(self):
        title=Text("Example on FlowLenia", font_size=36)
        title.to_edge(UP)
        self.play(FadeIn(title))
        self.wait(5)