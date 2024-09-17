from manim import *
import numpy as np

# Classe pour représenter un réseau complexe avec 5 nœuds disposés aléatoirement
class RandomPositionGRNNetwork(Scene):
    def construct(self):
        # Générer 5 positions aléatoires pour les nœuds (gènes)


        #Etcheverry Mayalen, Moulin-Frier Clément, Oudeyer Pierre-Yves, Levin Michael (2024) AI-driven Automated Discovery Tools Reveal Diverse Behavioral Competencies of Biological Networks eLife 13:RP92683
        #https://doi.org/10.7554/eLife.92683.3

        # Add this in bottom left corner
        reference = Text("(2024) AI-driven Automated Discovery Tools Reveal Diverse Behavioral Competencies of Biological Networks eLife 13:RP92683\nEtcheverry Mayalen, Moulin-Frier Clément, Oudeyer Pierre-Yves, Levin Michael",
                        font_size=12,  color=WHITE).to_corner(DL)
        self.add(reference)


        # add title: Example in real life: Gene Regulatory Network
        # added to the top of the screen
        title = Text("Example in real life: Gene Regulatory Networks",  font_size=24, color=WHITE).to_edge(UP)
        self.play(Write(title))

     #   self.wait(1)

        np. random.seed(43)  # Fixer la graine pour avoir des positions reproductibles

        positions=[[2.1903274, 0.55439924], [3.30034579, -1.45646228], [4.46283433, 2.05482495], [5.5, 0.14697327]]

        # shift y with 1.5
        for i in range(len(positions)):
            positions[i][1] -= .1


        # Créer les nœuds représentant les gènes avec des positions aléatoires
        def random_color():
            return np.random.uniform(0, 1, size=3)
        genes = [Dot(np.array([pos[0], pos[1], 0]), 
                        color=random_color(), ).scale(1.5) for pos in positions]

        # Ajouter des labels aux gènes
        labels = [
            Text(f"Gene {i+1}",
                 font_size=24,  color=WHITE
                 ).next_to(genes[i], RIGHT) for i in range(4)
        ]

        # Créer les arêtes (interactions) entre les gènes avec des flèches
        edges = [
            Arrow(genes[0].get_center(), genes[1].get_center(), color=WHITE),
            Arrow(genes[1].get_center(), genes[2].get_center(), color=WHITE),
            Arrow(genes[2].get_center(), genes[3].get_center(), color=WHITE),
            Arrow(genes[1].get_center(), genes[3].get_center(), color=WHITE),
        ]
        # write this in the left part of the screen, 
        # A gene regulatory network is a system of interacting genes and proteins that work together to control the activity of genes, determining how cells function and develop
        text = Text("A gene regulatory network is a system\nof interacting genes and proteins\nthat work together to control the activity of genes,\ndetermining how cells function and develop",
                     font_size=24, 
                     # interligne
                        line_spacing=1,
                     color=WHITE).to_edge(LEFT)
        self.play(*[Create(obj) for obj in [*genes, *labels, *edges]]    + [Write(text)])
        
       # self.wait(4)

        # clear the text
        self.play(FadeOut(text))


        # Small external perturbations lead to a large diversity of behaviors
        text = Text("Small external perturbations lead to\na large diversity of behaviors",
                     font_size=24, 
                     # interligne
                        line_spacing=1,
                     color=WHITE).to_edge(LEFT)
        self.play(Write(text))


        # add a red circle in the top left corner with wave effect
        perturbation=Dot(np.array([2.8, 1.5, 0]), color=RED).scale(2.5)

        self.play(FadeIn(perturbation))
        self.play(ApplyWave(perturbation))

        self.play(
            genes[0].animate.set_color(RED),
            genes[1].animate.set_color(RED),
            genes[2].animate.set_color(RED),
            genes[3].animate.set_color(RED),
        )

        # change perturbation to blue with wave effect
        perturbation.set_color(BLUE)
        self.play(ApplyWave(perturbation))

        self.play(
            genes[0].animate.set_color(BLUE),
            genes[1].animate.set_color(BLUE),
            genes[2].animate.set_color(BLUE),
            genes[3].animate.set_color(BLUE),
        )


        # change perturbation to green with wave effect
        perturbation.set_color(YELLOW)
        self.play(ApplyWave(perturbation))

        # randomize the color of the genes
        self.play(
            genes[0].animate.set_color(GREEN),
            genes[1].animate.set_color(RED),
            genes[2].animate.set_color(BLUE),
            genes[3].animate.set_color(YELLOW),
        )

        # anumate the external perturbations with a spark

        self.play(FadeOut(text))


        # But can we beat random search in perturbations space to explore this diversity?
        text = Text("But can we beat random search in\nperturbations space to explore this diversity?",
                     font_size=24, 
                     # interligne
                        line_spacing=1,
                     color=WHITE).to_edge(LEFT)
        self.play(Write(text))

        # fast multicolor perturbations of perturbation
        self.play( perturbation.animate.set_color(GREEN,   ))
        self.play( perturbation.animate.set_color(RED,  ))
        self.play( perturbation.animate.set_color(BLUE,   ))
        self.play( perturbation.animate.set_color(YELLOW,   ))
        self.play( perturbation.animate.set_color(PURPLE,    ))
        self.play( perturbation.animate.set_color(GOLD, ))
        self.play( perturbation.animate.set_color(PINK,  ))
        self.play( perturbation.animate.set_color(WHITE,     ))


        # insert paper_fig.mp4

        # fade out the text, perturbation and genes
        self.play(FadeOut(text), FadeOut(perturbation), *[FadeOut(obj) for obj in [*genes, *labels, *edges]])
    

        



 

        self.wait(4)


        # write IMGEP win 🏆
        # add a bottom text with the title of the paper
        text = Text("IMGEP win 🏆",
                     font_size=24, 
                     # interligne
                        line_spacing=1,
                     color=WHITE).to_edge(DOWN) 







        # Terminer l'animation avec un focus sur l'état final
        self.wait()

# Commande pour exécuter cette animation:
# manim -pql random_grn_network.py RandomPositionGRNNetwork
