from PySide6.QtWidgets import QPushButton, QStackedWidget, QLabel, QSlider, QComboBox
from PySide6.QtCore import Qt

from antra.interface.elements import SectionFrame, ValuePanel, ValueSlider, RangeSlider, WeightsPanel

class ControlsPanel(QStackedWidget):
    
    def __init__(self):
        super().__init__()
        self.controls_setup = self.build_setup()
        self.controls_select = self.build_select()
        self.controls_advice = self.build_advice()

        self.addWidget(self.controls_setup)
        self.addWidget(self.controls_select)
        self.addWidget(self.controls_advice)
    
    def build_setup(self) -> SectionFrame:
        frame = SectionFrame()

        # create and set buttons
        self.load_seg_btn    = QPushButton("Load Past Segmentation")
        self.import_ct_btn   = QPushButton("Import New Scan")
        self.load_config_btn = QPushButton("Load Past Settings")
        self.save_config_btn = QPushButton("Save Current Settings")
        self.start_seg_btn   = QPushButton("Start Segmentation")
        self.start_seg_btn.setDisabled(True)
        self.load_config_btn.setDisabled(True)
        self.save_config_btn.setDisabled(True)

        # add buttons
        frame.add_widget(self.load_seg_btn)
        frame.add_widget(self.import_ct_btn)
        frame.add_space(20)
        frame.add_widget(self.load_config_btn)
        frame.add_widget(self.save_config_btn)
        frame.add_space(50)
        frame.add_widget(self.start_seg_btn)

        return frame

    def build_select(self) -> SectionFrame:
        frame = SectionFrame()

        # mainly accesible labels for information
        self.toggle_seg_btn = QPushButton("Toggle Segmentations")
        self.confirm_tumor_btn = QPushButton("Confirm Selected Point")
        self.info_panel = ValuePanel("Tumor Information", frame)
        
        frame.add_widget(self.toggle_seg_btn)
        frame.add_widget(self.confirm_tumor_btn)
        frame.add_widget(self.info_panel)

        return frame

    def build_advice(self) -> SectionFrame:
        frame = SectionFrame()
        #  angular range sliders 
        frame.add_widget(QLabel("Theta range (azimuth):"))
        self.theta_slider = RangeSlider(0, 360, (60, 300), "°")
        frame.add_widget(self.theta_slider)

        frame.add_widget(QLabel("Phi range (elevation):"))
        self.phi_slider = RangeSlider(0, 180, (25, 155), "°")
        frame.add_widget(self.phi_slider)

        # density
        frame.add_widget(QLabel("Sampling density (rays/srad²):"))
        self.density_slider = ValueSlider(50, 2000, 1000)
        frame.add_widget(self.density_slider)
        frame.add_widget(QLabel("Max needle paths shown:"))
        self.max_results_slider = ValueSlider(1, 20, 5)
        frame.add_widget(self.max_results_slider)

        # start button 
        self.start_raytracing_btn = QPushButton("Start Raytracing")
        frame.add_widget(self.start_raytracing_btn)

        frame.add_space(16)

        # weights panel
        self.weights_panel = WeightsPanel()
        frame.add_widget(self.weights_panel)
        self.check_weights_btn = QPushButton("Rescore with Weights")
        self.check_weights_btn.setDisabled(True)
        frame.add_widget(self.check_weights_btn)

        frame.add_space(16)

        # advice selector
        self.find_paths_button = QPushButton("Show Adviced Needle Paths")
        frame.add_widget(self.find_paths_button)
        self.advice_combo = QComboBox()
        self.advice_combo.setCurrentText("Click to select path")
        self.advice_combo.setEnabled(False)
        frame.add_widget(self.advice_combo)

        # scoring color toggle 
        self.scoring_toggle_btn = QPushButton("Scoring Colors: ON")
        self.scoring_toggle_btn.setCheckable(True)
        self.scoring_toggle_btn.setChecked(True)
        self.scoring_toggle_btn.setEnabled(False)
        self.scoring_toggle_btn.toggled.connect(self._on_scoring_toggle)
        frame.add_widget(self.scoring_toggle_btn)

        self.phi_slider.changed.emit() # update slider values to basic settings


        return frame

    def populate_advice_combo(self, advice: list[dict]) -> None:
        '''Fill the combo box with ranked advice entries after raytracing.'''
        self.advice_combo.clear()
        for i, adv in enumerate(advice):
            self.advice_combo.addItem(f"#{i+1} Angle Margin: {adv['angle']:.1f} Score: {adv['score']:.3f}")
        self.advice_combo.setEnabled(True)
        self.scoring_toggle_btn.setEnabled(True)

    def _on_scoring_toggle(self, checked: bool):
        self.scoring_toggle_btn.setText("Scoring Colors: ON" if checked else "Scoring Colors: OFF")

    # helpers for reading current slider values
    def get_theta_rad(self) -> tuple[float, float]:
        return self.theta_slider.get_radians()

    def get_phi_rad(self) -> tuple[float, float]:
        return self.phi_slider.get_radians()

    def get_density(self) -> int:
        return self.density_slider.slider.value()

    def get_max_results(self) -> int:
        return self.max_results_slider.slider.value()