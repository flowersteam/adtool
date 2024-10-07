from manim import *
import numpy as np
import random

bars = []

class RandomPositionGRNNetwork(Scene):
    def construct(self):

        def update_gene_color(gene, value):
            if value >= 90:
                return gene.animate.set_color(GREEN)  # Change to green
            else:
                return gene.animate.set_color(RED)

        reference = Text("(2024) AI-driven Automated Discovery Tools Reveal Diverse Behavioral Competencies of Biological Networks eLife 13:RP92683\nEtcheverry Mayalen, Moulin-Frier Clément, Oudeyer Pierre-Yves, Levin Michael",
                        font_size=12,  color=WHITE).to_corner(DL)
        self.add(reference)

        title = Text("Example: Gene Regulatory Networks",  font_size=24, color=WHITE).to_edge(UP)
        self.play(Write(title))

        self.wait(1)

        np.random.seed(43)  # Fix seed for reproducible positions

        positions = [[2.1903274, 0.55439924], [3.30034579, -1.55646228], [4.46283433, 2.05482495], [5.5, -0.14697327]]

        for i in range(len(positions)):
            positions[i][1] -= .1

        genes = [Dot(np.array([pos[0], pos[1], 0]), color=WHITE).scale(1.5) for pos in positions]

        labels = [
            Text(f"Gene {i+1}",
                 font_size=24,  color=WHITE
                 ).next_to(genes[i], RIGHT) for i in range(4)
        ]

        edges = [
            Arrow(genes[0].get_center(), genes[1].get_center(), color=WHITE),
            Arrow(genes[1].get_center(), genes[2].get_center(), color=WHITE),
            Arrow(genes[2].get_center(), genes[3].get_center(), color=WHITE),
            Arrow(genes[1].get_center(), genes[3].get_center(), color=WHITE),
        ]

        def update_bar(bar, value):
            global bars
            new_height = np.interp(value, [10, 100], [0.2, 1.0])
            new_bar = Rectangle(width=0.2, height=new_height, fill_opacity=1)
            new_bar.next_to(genes[bars.index(bar)], LEFT, buff=0.1).shift(UP * (new_height / 2))
            color = interpolate_color(GREEN, RED, (value - 10) / 90)
            new_bar.set_fill(color)
            return Transform(bar, new_bar)
        
        def wave_perturbation(special_perturbation=None):
            global bars
            for t in range(9):
                animations = []
                animations.append(timer.animate.become(Text(f"T={t}", font_size=24, color=WHITE).move_to(np.array([2.8, 1.5, 0]))))
                for i in range(4):
                    if t == 0:
                        bar = Rectangle(width=0.2, height=0.5, fill_opacity=1)
                        bars.append(bar)
                        bar.next_to(genes[i], LEFT, buff=0.1).shift(UP * 0.25)  # Align bottom with the node
                        value = random.randint(10, 100)
                        color = interpolate_color(GREEN, RED, (value - 10) / 90)
                        bar.set_fill(color)
                        self.add(bar)
                    else:
                        # Special perturbation introduced on a specific gene (Gene 2)
                        if special_perturbation and i == special_perturbation['gene'] and t == 3:
                            value = special_perturbation['value']  # Set perturbation value
                            # add legend "small perturbation in gene activity introduced, led to a new trajectory" and a wave effect
                            animations.append(ApplyWave(genes[i], time_width=2))
                            # circumscribe the gene with a circle
                        #    animations.append(Circumscribe(genes[i]))
                        # flash
                            animations.append(Flash((genes[i]), color=RED))
                            # focus on
                     #       animations.append(FocusOn(genes[i]))
                
                        
                            animations.append(update_bar(bars[i], 100))
                       #     animations.append(ApplyWave(bars[i], time_width=2))
                        else:
                            value = random.randint(10, 100)
                            animations.append(update_bar(bars[i], value))


                self.play(*animations, run_time=0.75)

            # Clear all bars

        text = Text("A gene regulatory network is a dynamic system\nof interacting genes and proteins\nthat work together to control the activity of genes,\ndetermining how cells function and develop",
                    font_size=24, t2c={"dynamic": BLUE}, line_spacing=1, color=WHITE).to_edge(LEFT)

        self.play(*[Create(obj) for obj in [*genes, *labels, *edges]] + [Write(text)])

        timer = Text("T=0", font_size=24, color=WHITE).move_to(np.array([2.8, 1.5, 0]))
        self.add(timer)

        wave_perturbation()  # First run

    

        self.play(FadeOut(text))

        text = Text("Behavior space is defined as the final activity\nof genes of interest at the end of the simulation.\nExample: gene 1 and 2 activities",
                    font_size=24, line_spacing=1, color=WHITE,
                    t2c={"Behavior space": BLUE}).to_edge(LEFT)

        
        self.play(Write(text))

        self.play(*(
            #[ApplyWave(bars[i], time_width=2) for i in range(2)]+
  [Circumscribe(bars[i]) for i in range(2)]))
        
        self.wait(2)

        self.play(*[FadeOut(text), FadeOut(timer) ] )

        text1 = Text("Perturbations during development can be:",
                    font_size=24, line_spacing=1, color=WHITE,
                    t2c={"Perturbations": BLUE}
                    ).to_corner(LEFT).shift(2*UP)
        self.play(Write(text1)) 
        # write it below the previous text
        text2 = Text("1. Random noise added to gene activities",
                    font_size=24, line_spacing=1, color=WHITE).to_edge(LEFT).shift(UP)

        self.play(Write(text2))
        
        self.play(*([ update_bar(bars[i], 
                             min(100,   max(0, random.randint(-30, 30) 
                                 
                                 + bars[i].height*100 ))) for i in range(3)]+ [update_bar(bars[3], 20)]
             #     + [ApplyWave(bars[i], time_width=2) for i in range(4)]
                  ))
            
        # simulate noise addition to bars

        self.wait(1)

        text3 = Text("2. Sudden pushes to gene activities",
                    font_size=24, line_spacing=1, color=WHITE).to_edge(LEFT)
        
        self.play(Write(text3))
        self.play(update_bar(bars[3], 100) for i in range(4))

        self.wait(1)

        text4 = Text("3. Forbidden activations (states that are not allowed)",
                    font_size=24, line_spacing=1, color=WHITE).to_edge(LEFT).shift(DOWN)
        

        
        self.play(Write(text4)) #  update_bar(bars[3], 0)

        self.wait(1)
          # gene color change from red to green then red abruptly
      


        cross=Cross(stroke_color=RED, stroke_width=10).scale(0.1).move_to(genes[3].get_center())
       
        dot=Dot(
            color=WHITE
        ).scale(1.5).move_to(genes[3].get_center())
       
        self.play(  ReplacementTransform(genes[3], cross))
        self.play(*[update_bar(bars[3], 60),
        ]

                  , run_time=1)
        
        dot=Dot(
            color=GREEN
        ).scale(1.5).move_to(genes[3].get_center())
       
        
        # morph to dot
        self.play(*[
                    ReplacementTransform( cross,dot),
        ]
                    , run_time=1)        
        
        self.play(  update_bar(bars[3], 40))


        cross=Cross(stroke_color=RED, stroke_width=10).scale(0.1).move_to(genes[3].get_center())
       


        self.play(*[
                    ReplacementTransform( dot,cross),
        ]
                    , run_time=1)       
        self.play(*[update_bar(bars[3], 0),
                   
        ]
                    , run_time=1)




        self.wait(2)


        dot=Dot(
            color=WHITE
        ).scale(1.5).move_to(genes[3].get_center())

        genes[3]=dot
       

        self.play(*([FadeOut(obj) for obj in [text1, text2, text3, text4]]
                    + [ReplacementTransform(cross, dot)]))

        




        # Introduce second text for the perturbation
        text = Text("Starting from the same initial condition,\nsmall changes in gene activities during\ndevelopment lead to a wide range of new trajectories",
                    
                    t2c={"during": YELLOW},
                    
                    font_size=24, line_spacing=1, color=WHITE).to_edge(LEFT)
        
        self.play(Write(text))

        # ApplyWave to two bars
        self.wait(2) 



        # Introducing small perturbation to Gene 2 during the second run
        special_perturbation = {"gene": 1, "value": 75}  # Gene 2 (index 1), value 75

        timer.become(Text("T=0", font_size=24, color=WHITE).move_to(np.array([2.8, 1.5, 0])))

        self.play(*([FadeOut(bar) for bar in bars] ))
        bars.clear()

        # same but hide it after 2 seconds
        text2 = Paragraph("Example: One small perturbation in gene 2 activity introduced at T=3,\nled to a new trajectory, hence new areas of the behavior space",  
                    font_size=24, line_spacing=1, color=WHITE,
                    alignment="center",
                    t2c={"gene 2": YELLOW, "T=3": YELLOW,
                         "behavior space": BLUE
                         }
                    ).to_edge(DOWN).shift(UP * 0.5)
        self.play(Write(text2))


        self.wait(2)

        wave_perturbation(special_perturbation)  # Apply perturbation during the second run
        self.wait(1)  # Pause before repeating the wave

        self.play(*([FadeOut(bar) for bar in bars] + [FadeOut(timer)]))
        bars.clear()

        self.play(FadeOut(text), FadeOut(text2))

        # Final question text
        text = Text("But can we outperform random search\nwhen sampling the perturbation space\nto explore our behavior space?",
                    font_size=24, line_spacing=1, color=WHITE,
                    t2c={"outperform": YELLOW, "behavior space": BLUE
                         }
                    ).to_edge(LEFT)
        self.play(Write(text))

        self.wait(4)

        # Fade out everything before the conclusion
        self.play(FadeOut(text), *[FadeOut(obj) for obj in [*genes, *labels, *edges, *bars]])

        self.wait(4)
