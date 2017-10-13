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
            write(*,*) "2           ",net_dim%hl2
            write(*,*)

            write(*,*) "Dimension of feature space : ",D
            write(*,*) "Nonlinear function         : ",nlf
            write(*,*)

            write(*,*) "weights layer     shape"
            write(*,*) "1                 ",shape(net_weights%hl1)
            write(*,*) "2                 ",shape(net_weights%hl2)
            write(*,*) "3                 ",shape(net_weights%hl3)
        end subroutine info_net
        
        subroutine info_set(set_type)
            use config

            implicit none

            integer,intent(in) :: set_type

            !* scratch 
            integer :: ii,jj

            if (set_type.eq.1) then
                write(*,*) "========="
                write(*,*) "train set"
                write(*,*) "========="
                do ii=1,size(train_set),1
                    write(*,*) ""
                    write(*,*) "-------------"
                    write(*,*) "structure ",ii
                    write(*,*) "-------------"
                    write(*,*) "features:"
                    do jj=1,train_set(ii)%n
                        write(*,*) train_set(ii)%x(:,jj)
                    end do
                
                    write(*,*) ""
                    write(*,*) 'energy : ',train_set(ii)%energy
                    write(*,*) "forces:"
                    do jj=1,train_set(ii)%n
                        write(*,*) train_set(ii)%forces(:,jj)
                    end do

                end do
            else 
                write(*,*) "========"
                write(*,*) "test set"
                write(*,*) "========"
                do ii=1,size(test_set),1
                    write(*,*) ""
                    write(*,*) "-------------"
                    write(*,*) "structure ",ii
                    write(*,*) "-------------"
                    write(*,*) "features:"
                    do jj=1,test_set(ii)%n
                        write(*,*) test_set(ii)%x(:,jj)
                    end do
                
                    write(*,*) ""
                    write(*,*) 'energy : ',test_set(ii)%energy
                    write(*,*) "forces:"
                    do jj=1,test_set(ii)%n
                        write(*,*) test_set(ii)%forces(:,jj)
                    end do

                end do
            end if
        end subroutine info_set
end module io
