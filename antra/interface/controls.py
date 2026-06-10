import easygui

from PySide6.QtWidgets import QPushButton, QStackedWidget

from antra.interface.elements import SectionFrame


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

        # buttons
        self.load_seg_btn      = QPushButton("Load Past Segmentation")
        self.import_ct_btn     = QPushButton("Import New Scan")
        self.load_config_btn   = QPushButton("Load Past Settings")
        self.save_config_btn   = QPushButton("Save Current Settings")
        self.start_seg_btn     = QPushButton("Start Segmentation")
        self.start_seg_btn.setDisabled(True)
        self.load_config_btn.setDisabled(True)
        self.save_config_btn.setDisabled(True)

        # config


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
        return frame

    def build_advice(self) -> SectionFrame:
        frame = SectionFrame()
        return frame