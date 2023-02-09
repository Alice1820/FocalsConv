from .detector3d_template import Detector3DTemplate


class PVRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        # print (batch_dict.keys())
        # exit()
        # print (batch_dict['batch_box_preds'])
        # print (batch_dict['batch_box_preds'].size())
        # batch_dict['batch_box_preds'].mean().backward()
        # '''assert gradients'''
        # for name, param in self.named_parameters():
        #     print(name, param.grad)
        #     exit()
        # exit()
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            if 'loss_box_of_pts' in batch_dict:
                loss += batch_dict['loss_box_of_pts']
                tb_dict['loss_box_of_pts'] = batch_dict['loss_box_of_pts']
                
            ret_dict = {
                'loss': loss
            }
            # pred_dicts, recall_dicts = self.post_processing(batch_dict)
            # return pred_dicts, recall_dicts
            return ret_dict, tb_dict, disp_dict
            # return ret_dict, tb_dict, disp_dict, pred_dicts, recall_dicts
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict

    # def get_semi_loss(self):
    #     loss_rpn_2D = 
