module io
    implicit none

    contains
        subroutine error(routine,message)
            implicit none

            character(len=*),intent(in) :: routine,message

            write(*,*) '\n****************************'
            write(*,*) 'error raised in routine',routine,':'
            write(*,*) message
            write(*,*) '****************************\n'
            call exit(0)
        end subroutine

        subroutine info_net()
            use config

            implicit none

            write(*,*) "NN layer    number of nodes"
            write(*,*) "1           ",net_dim%hl1
            write(*,*) "2           ",net_dim%hl2,"\n"

            write(*,*) "Dimension of feature space : ",D
            write(*,*) "Nonlinear function         : ",nlf,"\n"

            write(*,*) "weights layer     shape"
            write(*,*) "1                 ",shape(net_weights%hl1)
            write(*,*) "2                 ",shape(net_weights%hl2)
            write(*,*) "3                 ",shape(net_weights%hl3)
        end subroutine info_net
end module io
