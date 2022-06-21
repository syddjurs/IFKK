import lxml.etree as ET
import pandas as pd


class GenericParser:
    def __init__(self):
        self.block = []
        self.frame = None

    def parse(self, xml_chunk):
        for element in ET.fromstring(xml_chunk).findall(self.find):
            element_dict = {}
            for child in element.iter():
                if child.tag and child.text:
                    parent_name = child.getparent().tag.split("}")[-1]
                    attrib_name = child.tag.split("}")[-1]
                    if parent_name == self.find.split("}")[-1]:
                        name = attrib_name
                    else:
                        name = f"{parent_name}_{attrib_name}"
                    element_dict[name] = child.text
            if element_dict:
                self.block.append(element_dict)
        self.frame = pd.DataFrame(self.block)

    def __str__(self):
        return self.frame.to_string()


class Trackers(GenericParser):
    def __init__(self):
        self.find = ".//{http://schemas.datacontract.org/2004/07/PublicSoapApi.DTO.Model}DTTracker"
        super().__init__()


class DrivingBook(GenericParser):
    def __init__(self):
        self.find = ".//{http://schemas.datacontract.org/2004/07/PublicSoapApi.DTO.Model}DTMileageLog"
        super().__init__()


class MileageLogPositions(GenericParser):
    def __init__(self):
        self.find = ".//{http://schemas.datacontract.org/2004/07/PublicSoapApi.DTO.Model}DTGpsPosition"
        super().__init__()
